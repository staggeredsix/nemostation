from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import gradio as gr
import requests

from src.demo.orchestrator import run_demo_stream

STATIC_DIR = Path(__file__).resolve().parent / "static"
CSS_PATH = STATIC_DIR / "style.css"

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")


def ping_server() -> bool:
    try:
        resp = requests.get(f"{SERVER_URL}/v1/models", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def badge(status: str) -> str:
    return f"<span class='badge {status}'>{status.title()}</span>"


def render_progress(stages: List[Dict]) -> str:
    done = len([s for s in stages if s["status"] == "done"])
    total = len(stages)
    pct = int((done / total) * 100) if total else 0
    return f"<div class='progress-shell'><div class='progress-fill' style='width:{pct}%;'></div></div>"


def render_timeline(state: Dict) -> str:
    rows = []
    for stage in state.get("stages", []):
        rows.append(
            f"<div class='timeline-row'>"
            f"<div class='timeline-name'>{stage['name']}</div>"
            f"<div>{badge(stage['status'])}</div>"
            f"<div>{stage['ms']:.0f} ms</div>"
            f"<div>~{stage['ttft_ms']:.0f} ms</div>"
            f"<div>~{stage['tok_s']:.1f} tok/s</div>"
            f"<div>{stage.get('tokens', 0)} tok</div>"
            f"</div>"
        )
    return "".join(rows)


def render_metrics(state: Dict) -> str:
    metrics = state.get("metrics", {})
    completed = len([s for s in state.get("stages", []) if s["status"] == "done"])
    total = len(state.get("stages", []))
    return f"""
    <div class='metrics-grid'>
      <div class='metrics-card'><div class='label'>Total time</div><div class='value'>{metrics.get('total_ms',0):.0f} ms</div></div>
      <div class='metrics-card'><div class='label'>Approx TTFT</div><div class='value'>~{metrics.get('approx_ttft_ms',0):.0f} ms</div></div>
      <div class='metrics-card'><div class='label'>Approx tok/s</div><div class='value'>~{metrics.get('approx_tok_s',0):.1f}</div></div>
      <div class='metrics-card'><div class='label'>Stages complete</div><div class='value'>{completed}/{total}</div></div>
    </div>
    """


def render_agent_outputs(state: Dict) -> Dict[str, str]:
    outputs = {}
    for stage in state.get("stages", []):
        outputs[stage["name"].lower()] = stage.get("output", "") or stage.get("error", "")
    return outputs


def stream_runner(goal: str, scenario: str, fast: bool):
    if not ping_server():
        raise gr.Error("Server not ready at /v1/models. Start run_server.sh first.")

    for state in run_demo_stream(goal, fast=fast, scenario=scenario or None):
        metrics_html = render_metrics(state)
        timeline_html = render_progress(state["stages"]) + render_timeline(state)
        outputs = render_agent_outputs(state)
        final_text = state.get("final", "")
        yield (
            metrics_html,
            timeline_html,
            outputs.get("planner", ""),
            outputs.get("coder", ""),
            outputs.get("reviewer", ""),
            outputs.get("ops", ""),
            outputs.get("aggregator", ""),
            final_text,
        )


def build_ui() -> gr.Blocks:
    css = CSS_PATH.read_text()
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Nemotron-3 Nano Agentic Demo — HF + vLLM")
        with gr.Row():
            with gr.Column(scale=1):
                goal = gr.Textbox(label="Goal", value="Create a demo plan for a local agent stack", lines=3)
                scenario = gr.Dropdown(
                    label="Scenario",
                    choices=[
                        "Optimize inference for latency",
                        "Ship a resilient offline demo",
                        "Benchmark throughput on GB300",
                    ],
                    value="Optimize inference for latency",
                )
                fast = gr.Checkbox(label="Fast mode (skip Ops, fewer tokens)")
                run_btn = gr.Button("Run Demo", variant="primary")
                server_status = gr.HTML("", label="Server status")
            with gr.Column(scale=2):
                with gr.Tab("Timeline"):
                    metrics_card = gr.HTML()
                    timeline = gr.HTML(elem_classes=["card"])
                with gr.Tab("Agent Outputs"):
                    planner_box = gr.Textbox(label="Planner", lines=6, show_copy_button=True)
                    coder_box = gr.Textbox(label="Coder", lines=6, show_copy_button=True)
                    reviewer_box = gr.Textbox(label="Reviewer", lines=6, show_copy_button=True)
                    ops_box = gr.Textbox(label="Ops", lines=6, show_copy_button=True)
                    aggregator_box = gr.Textbox(label="Aggregator", lines=6, show_copy_button=True)
                with gr.Tab("Final Answer"):
                    final_box = gr.Textbox(label="Final", lines=8, show_copy_button=True)

        def update_status():
            status = ping_server()
            pill_class = "up" if status else "down"
            text = "Online" if status else "Offline"
            return f"<div class='server-pill {pill_class}'>● Server {text}</div>"

        demo.load(fn=update_status, inputs=None, outputs=server_status)

        run_btn.click(
            fn=stream_runner,
            inputs=[goal, scenario, fast],
            outputs=[
                metrics_card,
                timeline,
                planner_box,
                coder_box,
                reviewer_box,
                ops_box,
                aggregator_box,
                final_box,
            ],
            show_progress=True,
        )

    return demo


def main():
    demo = build_ui()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.queue()
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=port, share=False)


if __name__ == "__main__":
    main()
