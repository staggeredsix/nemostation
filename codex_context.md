# Codex Agent: Nemotron-3 Nano Agentic Demo (HF-only)

## Objective
Create a production-ready demo repository that runs NVIDIA Nemotron-3 Nano locally
from Hugging Face and demonstrates agentic reasoning (planner / coder / reviewer / ops)
using a single model with role-conditioned prompts.

This must be stable, simple, and runnable on a single high-end GPU system (GB300).

---

## Model (official, available now)
Default model:
- nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Requirements:
- Use Hugging Face weights directly
- No NVIDIA NIM
- No managed services
- No distributed setup

Allow switching models via env var:
- MODEL_ID (default to BF16 above)
- Optional FP8 variant if available

---

## Serving Stack
Use **vLLM** with OpenAI-compatible API.

Constraints:
- Single node
- Conservative defaults (stability > peak throughput)
- Explicit max context length (do NOT assume 1M tokens)

Target API:
- http://localhost:8000/v1

---

## Repository Layout (must match exactly)

nemotron-agent-demo/
├── README.md
├── requirements.txt
├── run_server.sh
├── run_demo.sh
├── context/
│   └── gb300.json
├── prompts/
│   ├── system.txt
│   ├── planner.txt
│   ├── coder.txt
│   ├── reviewer.txt
│   ├── ops.txt
│   └── aggregator.txt
└── src/
    ├── server/
    │   └── healthcheck.py
    └── demo/
        └── run_demo.py

---

## requirements.txt
Include only:
- vllm
- transformers
- openai
- requests
- rich (optional, for formatting)

Do NOT include CUDA, torch install commands, or drivers.

---

## run_server.sh
Responsibilities:
- Read MODEL_ID from env (default to Nemotron-3 Nano BF16)
- Launch vLLM OpenAI server
- Set safe defaults:
  - max_model_len ≈ 16384
  - gpu-memory-utilization ≈ 0.9
- Block until healthcheck passes
- Print “SERVER READY”

Server must be killable with Ctrl-C.

---

## healthcheck.py
- Poll http://localhost:8000/v1/models
- Exit 0 on success
- Exit non-zero on timeout (30s max)

---

## Prompts (core of the demo)

### system.txt
Establish:
- You are NVIDIA Nemotron-3 Nano
- Running on a GB300-class GPU
- Prioritize correctness, structure, and performance
- Prefer concise, actionable outputs

### planner.txt
Role: Planner
- Break task into steps
- Assign work to agents
- Output structured plan

### coder.txt
Role: Coder
- Produce minimal, correct code or config
- Avoid speculation
- Assume GB300-class hardware

### reviewer.txt
Role: Reviewer
- Identify flaws
- Call out incorrect assumptions
- Propose concrete fixes

### ops.txt
Role: Ops
- Optimize for inference performance
- Suggest batching, memory, runtime flags
- Note failure modes

### aggregator.txt
Role: Aggregator
- Merge all agent outputs
- Resolve conflicts
- Produce final authoritative answer
- Include risks + mitigations

---

## Hardware Context Injection
Create context/gb300.json with static system facts:
- GPU model
- Memory size
- NVLink availability
- Intended workload

Inject this as a SYSTEM message before agent prompts.

---

## Demo Runner (run_demo.py)

Flow:
1. Prompt user for a goal
2. Call Planner agent
3. Call Coder, Reviewer, Ops agents sequentially
4. Call Aggregator with all prior outputs
5. Print clearly labeled sections

Rules:
- One agent = one LLM call
- Same model for all agents
- Temperature ≈ 0.2
- No external tools
- No frameworks (LangChain, CrewAI, etc.)

Add --fast flag:
- Skips Ops agent
- Reduces max tokens

---

## run_demo.sh
- Verify server health
- Run demo script
- Fail fast with clear error if server unavailable

---

## README.md (must include)
- What the demo shows (agentic reasoning illusion)
- Setup steps (venv + pip install)
- How to start server
- How to run demo
- How to reduce memory usage
- Common failures + fixes

---

## Acceptance Criteria
- One-command server startup
- One-command demo execution
- No NIM usage
- Works offline after model download
- Clean, readable output suitable for live demos

---

## Engineering Priorities (in order)
1. Stability
2. Clarity
3. Demo impact
4. Performance tuning (last)

Do not over-optimize. Do not over-engineer.

Deliver a repo that “just works”.
