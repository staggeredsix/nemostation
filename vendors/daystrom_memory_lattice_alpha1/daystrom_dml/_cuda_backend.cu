#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err = (expr);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err));                 \
        }                                                                       \
    } while (0)

__global__ void compute_norms(const float* data, float* norms, int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        float v = data[row * dim + idx];
        sum += v * v;
    }
    __shared__ float shared[256];
    int lane = threadIdx.x;
    shared[lane] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        norms[row] = sqrtf(shared[0]);
    }
}

__global__ void cosine_kernel(const float* queries,
                              const float* keys,
                              const float* query_norms,
                              const float* key_norms,
                              float* output,
                              int q_rows,
                              int k_rows,
                              int dim) {
    int q_idx = blockIdx.x;
    int k_idx = blockIdx.y;
    if (q_idx >= q_rows || k_idx >= k_rows) return;

    float q_norm = query_norms[q_idx];
    float k_norm = key_norms[k_idx];
    if (q_norm == 0.0f || k_norm == 0.0f) {
        if (threadIdx.x == 0) {
            output[q_idx * k_rows + k_idx] = 0.0f;
        }
        return;
    }

    const float* q_ptr = queries + q_idx * dim;
    const float* k_ptr = keys + k_idx * dim;
    float dot = 0.0f;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        dot += q_ptr[idx] * k_ptr[idx];
    }

    __shared__ float shared[256];
    int lane = threadIdx.x;
    shared[lane] = dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[q_idx * k_rows + k_idx] = shared[0] / (q_norm * k_norm);
    }
}

py::array_t<float> cosine_sim_matrix(
    py::array_t<float, py::array::c_style | py::array::forcecast> queries,
    py::array_t<float, py::array::c_style | py::array::forcecast> keys) {
    if (queries.ndim() != 2 || keys.ndim() != 2) {
        throw std::invalid_argument("queries and keys must be 2D arrays");
    }
    if (queries.shape(1) != keys.shape(1)) {
        throw std::invalid_argument("Query and key dimensions must match");
    }
    const int q_rows = static_cast<int>(queries.shape(0));
    const int k_rows = static_cast<int>(keys.shape(0));
    const int dim = static_cast<int>(queries.shape(1));

    auto result = py::array_t<float>({q_rows, k_rows});
    auto queries_buf = queries.request();
    auto keys_buf = keys.request();
    auto result_buf = result.request();

    const float* h_queries = static_cast<float*>(queries_buf.ptr);
    const float* h_keys = static_cast<float*>(keys_buf.ptr);
    float* h_output = static_cast<float*>(result_buf.ptr);

    float *d_queries = nullptr, *d_keys = nullptr, *d_output = nullptr;
    float *d_q_norms = nullptr, *d_k_norms = nullptr;

    size_t q_bytes = static_cast<size_t>(q_rows * dim) * sizeof(float);
    size_t k_bytes = static_cast<size_t>(k_rows * dim) * sizeof(float);
    size_t out_bytes = static_cast<size_t>(q_rows * k_rows) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_queries, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_keys, k_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_q_norms, q_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_norms, k_rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_queries, h_queries, q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, k_bytes, cudaMemcpyHostToDevice));

    dim3 norm_grid_q(q_rows);
    dim3 norm_grid_k(k_rows);
    dim3 norm_block(256);
    compute_norms<<<norm_grid_q, norm_block>>>(d_queries, d_q_norms, q_rows, dim);
    compute_norms<<<norm_grid_k, norm_block>>>(d_keys, d_k_norms, k_rows, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 grid(q_rows, k_rows);
    dim3 block(256);
    size_t shared_mem = block.x * sizeof(float);
    cosine_kernel<<<grid, block, shared_mem>>>(d_queries, d_keys, d_q_norms, d_k_norms,
                                               d_output, q_rows, k_rows, dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_queries);
    cudaFree(d_keys);
    cudaFree(d_output);
    cudaFree(d_q_norms);
    cudaFree(d_k_norms);

    return result;
}

std::pair<py::array_t<int64_t>, py::array_t<float>> top_k(
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    int k) {
    if (scores.ndim() != 2) {
        throw std::invalid_argument("scores must be a 2D array");
    }
    const int rows = static_cast<int>(scores.shape(0));
    const int cols = static_cast<int>(scores.shape(1));
    if (k <= 0) {
        auto idx_out = py::array_t<int64_t>({rows, 0});
        auto score_out = py::array_t<float>({rows, 0});
        return {idx_out, score_out};
    }
    k = std::min(k, cols);

    auto idx_out = py::array_t<int64_t>({rows, k});
    auto score_out = py::array_t<float>({rows, k});
    auto scores_buf = scores.request();

    const float* h_scores = static_cast<float*>(scores_buf.ptr);
    thrust::device_vector<float> d_scores(h_scores, h_scores + (rows * cols));
    thrust::device_vector<int64_t> d_indices(cols);

    auto idx_buf = idx_out.request();
    auto score_buf = score_out.request();
    int64_t* h_indices = static_cast<int64_t*>(idx_buf.ptr);
    float* h_top_scores = static_cast<float*>(score_buf.ptr);

    for (int row = 0; row < rows; ++row) {
        auto row_begin = d_scores.begin() + static_cast<size_t>(row) * cols;
        auto row_end = row_begin + cols;
        d_indices.resize(cols);
        thrust::sequence(d_indices.begin(), d_indices.end());
        thrust::sort_by_key(row_begin, row_end, d_indices.begin(), thrust::greater<float>());

        thrust::copy_n(d_indices.begin(), k, h_indices + static_cast<size_t>(row) * k);
        thrust::copy_n(row_begin, k, h_top_scores + static_cast<size_t>(row) * k);
    }

    return {idx_out, score_out};
}

PYBIND11_MODULE(_cuda_backend, m) {
    m.doc() = "CUDA-accelerated primitives for Daystrom DML";
    m.def("cosine_sim_matrix", &cosine_sim_matrix, "Compute cosine similarity matrix");
    m.def("top_k", &top_k, "Select top-k scores per row");
}
