from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import gradio as gr
import requests

from src.demo.orchestrator import run_demo_stream
from src.demo.prompts import (
    clear_override,
    delete_goal_preset,
    diff_summary,
    get_active_prompt,
    get_saved_override,
    load_default_prompts,
    load_goal_presets,
    set_active_override,
    upsert_goal_preset,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
CSS_PATH = STATIC_DIR / "style.css"

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://nemotron-server:8000/v1").rstrip("/")


def ping_server() -> bool:
    try:
        resp = requests.get(f"{OPENAI_BASE_URL}/models", timeout=3)
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
    dml_info = state.get("dml", {})
    dml_enabled = bool(dml_info.get("enabled"))
    dml_requested = bool(dml_info.get("requested"))
    dml_top_k = dml_info.get("top_k", 0)
    dml_error = dml_info.get("error")
    if dml_requested and not dml_enabled and dml_error:
        dml_label = f"DML: OFF ({dml_error})"
        dml_class = "dml-pill off"
    elif dml_enabled:
        dml_label = f"DML: ON (k={dml_top_k})"
        dml_class = "dml-pill on"
    else:
        dml_label = "DML: OFF"
        dml_class = "dml-pill off"

    rows = []
    rows.append(f"<div class='{dml_class}'>{dml_label}</div>")
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


def _memory_key(entry: Dict) -> str:
    meta = entry.get("meta") or {}
    return str(meta.get("doc_path") or entry.get("id") or "unknown")


def _memory_score(entry: Dict) -> str:
    for key in ("fidelity", "score", "semantic_score"):
        value = entry.get(key)
        if value is not None:
            try:
                return f"{float(value):.3f}"
            except (TypeError, ValueError):
                return str(value)
    return ""


def _render_memory_table(entries: List[Dict], latency_ms: int, token_estimate: int, error: str | None) -> str:
    if error:
        return f"<div class='memory-error'>DML error: {error}</div>"
    if not entries:
        return "<div class='memory-empty'>No memories retrieved.</div>"
    header = ""
    if latency_ms or token_estimate:
        header = (
            "<div class='memory-meta'>"
            f"Latency: {latency_ms} ms · Token estimate: {token_estimate}"
            "</div>"
        )
    rows = []
    for entry in entries:
        meta = entry.get("meta") or {}
        doc_path = meta.get("doc_path", "")
        memory_id = entry.get("id", "")
        summary = entry.get("summary", "")
        score = _memory_score(entry)
        rows.append(
            "<tr>"
            f"<td>{memory_id}</td>"
            f"<td>{doc_path}</td>"
            f"<td>{summary}</td>"
            f"<td>{score}</td>"
            "</tr>"
        )
    table = (
        "<table class='memory-table'>"
        "<thead><tr><th>Memory ID</th><th>Doc Path</th><th>Summary</th><th>Fidelity/Score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    return header + table


def render_memory_access(state: Dict) -> Dict[str, str]:
    per_stage = {}
    all_entries: Dict[str, Dict] = {}
    for stage in state.get("stages", []):
        dml_info = stage.get("dml") or {}
        entries = dml_info.get("entries", [])
        for entry in entries:
            key = _memory_key(entry)
            if key not in all_entries:
                all_entries[key] = entry
        per_stage[stage["name"].lower()] = _render_memory_table(
            entries,
            int(dml_info.get("latency_ms", 0)),
            int(dml_info.get("token_estimate", 0)),
            dml_info.get("error"),
        )

    combined_entries = list(all_entries.values())
    combined_table = _render_memory_table(combined_entries, 0, 0, None)
    return {
        "all": combined_table,
        "planner": per_stage.get("planner", ""),
        "coder": per_stage.get("coder", ""),
        "reviewer": per_stage.get("reviewer", ""),
        "ops": per_stage.get("ops", ""),
        "aggregator": per_stage.get("aggregator", ""),
    }


def _default_prompt_source() -> str:
    return "<div class='prompt-source'><span class='label'>Prompt source:</span> typed</div>"


def stream_runner(goal: str, scenario: str, fast: bool, use_dml: bool, dml_top_k: int):
    if not ping_server():
        raise gr.Error("Server not ready at /v1/models. Start docker compose first.")

    for state in run_demo_stream(goal, fast=fast, scenario=scenario or None, use_dml=use_dml, dml_top_k=dml_top_k):
        metrics_html = render_metrics(state)
        timeline_html = render_progress(state["stages"]) + render_timeline(state)
        outputs = render_agent_outputs(state)
        final_text = state.get("final", "")
        memory_panels = render_memory_access(state)
        yield (
            metrics_html,
            timeline_html,
            outputs.get("planner", ""),
            outputs.get("coder", ""),
            outputs.get("reviewer", ""),
            outputs.get("ops", ""),
            outputs.get("aggregator", ""),
            final_text,
            memory_panels["all"],
            memory_panels["planner"],
            memory_panels["coder"],
            memory_panels["reviewer"],
            memory_panels["ops"],
            memory_panels["aggregator"],
        )


def build_ui() -> gr.Blocks:
    css = CSS_PATH.read_text()
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Nemotron-3 Nano Agentic Demo — HF + Transformers")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    goal = gr.Textbox(label="Goal", value="Create a demo plan for a local agent stack", lines=3, scale=4)
                    prompt_source = gr.HTML(value=_default_prompt_source(), scale=1)
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
                use_dml = gr.Checkbox(label="DML Memory ON / OFF")
                dml_top_k = gr.Slider(label="DML top_k", minimum=1, maximum=10, step=1, value=6, visible=False)
                run_btn = gr.Button("Run Demo", variant="primary")
                server_status = gr.HTML("", label="Server status")
            with gr.Column(scale=2):
                with gr.Tab("Timeline"):
                    metrics_card = gr.HTML()
                    timeline = gr.HTML(elem_classes=["card"])
                with gr.Tab("Agent Outputs"):
                    planner_box = gr.Textbox(label="Planner", lines=6)
                    coder_box = gr.Textbox(label="Coder", lines=6)
                    reviewer_box = gr.Textbox(label="Reviewer", lines=6)
                    ops_box = gr.Textbox(label="Ops", lines=6)
                    aggregator_box = gr.Textbox(label="Aggregator", lines=6)
                with gr.Tab("Final Answer"):
                    final_box = gr.Textbox(label="Final", lines=8)
                with gr.Tab("Memory Access"):
                    memory_all = gr.HTML(label="All accessed memories")
                    with gr.Accordion("Planner"):
                        memory_planner = gr.HTML()
                    with gr.Accordion("Coder"):
                        memory_coder = gr.HTML()
                    with gr.Accordion("Reviewer"):
                        memory_reviewer = gr.HTML()
                    with gr.Accordion("Ops"):
                        memory_ops = gr.HTML()
                    with gr.Accordion("Aggregator"):
                        memory_aggregator = gr.HTML()

        with gr.Tab("Prompts"):
            with gr.Tab("Goal Presets"):
                presets_state = gr.State(load_goal_presets())
                with gr.Row():
                    preset_dropdown = gr.Dropdown(label="Preset", choices=[p["name"] for p in presets_state.value], value=presets_state.value[0]["name"] if presets_state.value else None)
                    unsaved_indicator = gr.HTML(value="")
                preset_content = gr.Textbox(label="Preset content", lines=5)
                preset_status = gr.Markdown(value="")
                with gr.Row():
                    push_goal_btn = gr.Button("Push to Goal Input", variant="primary")
                    save_preset_btn = gr.Button("Save Preset")
                    save_new_btn = gr.Button("Save as New")
                with gr.Row():
                    new_name = gr.Textbox(label="New preset name", placeholder="Enter name for new preset")
                    delete_confirm = gr.Checkbox(label="Confirm delete")
                    delete_btn = gr.Button("Delete Preset", variant="stop")

            with gr.Tab("Agent Prompts"):
                agent_dropdown = gr.Dropdown(
                    label="Agent",
                    choices=["system", "planner", "coder", "reviewer", "ops", "aggregator"],
                    value="system",
                )
                badge_holder = gr.HTML(value="")
                diff_text = gr.Markdown(value="")
                with gr.Row():
                    default_prompt = gr.Textbox(label="Default Prompt (read-only)", lines=10, interactive=False)
                    active_prompt = gr.Textbox(label="Active Prompt (editable)", lines=10, interactive=True)
                agent_status = gr.Markdown(value="")
                with gr.Row():
                    use_default_btn = gr.Button("Use Default")
                    save_override_btn = gr.Button("Save Override", variant="primary")
                    push_live_btn = gr.Button("Push Override Live")

        def update_goal_source(_: str) -> str:
            return _default_prompt_source()

        def _preset_payload(target_name: str | None = None, status: str = "", clear_name: bool = False):
            presets = load_goal_presets()
            names = [p["name"] for p in presets]
            selected = target_name if target_name in names else (names[0] if names else None)
            content = next((p["content"] for p in presets if p["name"] == selected), "") if selected else ""
            new_name_value = "" if clear_name else gr.update()
            return (
                presets,
                gr.update(choices=names, value=selected),
                gr.update(value=content),
                "",
                status,
                new_name_value,
            )

        def load_presets():
            return _preset_payload()

        def select_preset(name: str):
            return _preset_payload(target_name=name)

        def save_preset(name: str, content: str):
            if not name:
                raise gr.Error("Choose a preset before saving.")
            upsert_goal_preset(name, content)
            return _preset_payload(target_name=name, status=f"Saved preset **{name}**.", clear_name=True)

        def save_new_preset(name: str, content: str):
            if not name:
                raise gr.Error("Enter a name to save a new preset.")
            upsert_goal_preset(name, content)
            return _preset_payload(target_name=name, status=f"Saved new preset **{name}**.", clear_name=True)

        def delete_preset(name: str, confirm: bool):
            if not name:
                raise gr.Error("No preset selected.")
            if not confirm:
                raise gr.Error("Check confirm delete before removing a preset.")
            delete_goal_preset(name)
            return _preset_payload(status=f"Deleted preset **{name}**.", clear_name=True)

        def check_unsaved(content: str, name: str):
            presets = load_goal_presets()
            saved = next((p["content"] for p in presets if p["name"] == name), "") if name else ""
            if not name:
                return ""
            return "<div class='unsaved'>Unsaved changes</div>" if content.strip() != saved.strip() else ""

        def push_goal(content: str, name: str):
            source = f"<div class='prompt-source'><span class='label'>Prompt source:</span> preset:{name}</div>" if name else _default_prompt_source()
            return gr.update(value=content), source

        def _agent_badge(agent: str, active: str, default: str) -> str:
            overridden = active.strip() != default.strip() or bool(get_saved_override(agent))
            label = "OVERRIDDEN" if overridden else "DEFAULT"
            css_class = "override" if overridden else "default"
            return f"<div class='prompt-badge {css_class}'>{label}</div>"

        def load_agent(agent: str):
            defaults = load_default_prompts()
            default_value = defaults.get(agent, "")
            active_value = get_active_prompt(agent)
            badge_html = _agent_badge(agent, active_value, default_value)
            return (
                gr.update(value=default_value),
                gr.update(value=active_value),
                badge_html,
                diff_summary(agent, active_value),
                "",
            )

        def set_override(agent: str, content: str, persist: bool, label: str):
            set_active_override(agent, content, persist=persist)
            defaults = load_default_prompts()
            default_value = defaults.get(agent, "")
            badge_html = _agent_badge(agent, content, default_value)
            status = f"{label} for **{agent}**." + (" (Session-only)" if not persist else "")
            return (
                gr.update(value=default_value),
                gr.update(value=content),
                badge_html,
                diff_summary(agent, content),
                status,
            )

        def reset_to_default(agent: str):
            clear_override(agent, persist=True)
            defaults = load_default_prompts()
            default_value = defaults.get(agent, "")
            badge_html = _agent_badge(agent, default_value, default_value)
            return (
                gr.update(value=default_value),
                gr.update(value=default_value),
                badge_html,
                diff_summary(agent, default_value),
                f"Reverted {agent} to default.",
            )

        def update_diff(content: str, agent: str):
            defaults = load_default_prompts()
            default_value = defaults.get(agent, "")
            badge_html = _agent_badge(agent, content, default_value)
            return badge_html, diff_summary(agent, content)

        def update_status():
            status = ping_server()
            pill_class = "up" if status else "down"
            text = "Online" if status else "Offline"
            return f"<div class='server-pill {pill_class}'>● Server {text}</div>"

        demo.load(fn=update_status, inputs=None, outputs=server_status)

        demo.load(fn=load_presets, inputs=None, outputs=[presets_state, preset_dropdown, preset_content, unsaved_indicator, preset_status, new_name])
        demo.load(fn=load_agent, inputs=agent_dropdown, outputs=[default_prompt, active_prompt, badge_holder, diff_text, agent_status])

        preset_dropdown.change(fn=select_preset, inputs=preset_dropdown, outputs=[presets_state, preset_dropdown, preset_content, unsaved_indicator, preset_status, new_name])
        preset_content.change(fn=check_unsaved, inputs=[preset_content, preset_dropdown], outputs=unsaved_indicator)
        save_preset_btn.click(fn=save_preset, inputs=[preset_dropdown, preset_content], outputs=[presets_state, preset_dropdown, preset_content, unsaved_indicator, preset_status, new_name])
        save_new_btn.click(fn=save_new_preset, inputs=[new_name, preset_content], outputs=[presets_state, preset_dropdown, preset_content, unsaved_indicator, preset_status, new_name])
        delete_btn.click(fn=delete_preset, inputs=[preset_dropdown, delete_confirm], outputs=[presets_state, preset_dropdown, preset_content, unsaved_indicator, preset_status, new_name])
        push_goal_btn.click(fn=push_goal, inputs=[preset_content, preset_dropdown], outputs=[goal, prompt_source])

        goal.input(fn=update_goal_source, inputs=goal, outputs=prompt_source)

        agent_dropdown.change(fn=load_agent, inputs=agent_dropdown, outputs=[default_prompt, active_prompt, badge_holder, diff_text, agent_status])
        active_prompt.change(fn=update_diff, inputs=[active_prompt, agent_dropdown], outputs=[badge_holder, diff_text])
        save_override_btn.click(fn=set_override, inputs=[agent_dropdown, active_prompt, gr.State(True), gr.State("Saved override")], outputs=[default_prompt, active_prompt, badge_holder, diff_text, agent_status])
        push_live_btn.click(fn=set_override, inputs=[agent_dropdown, active_prompt, gr.State(False), gr.State("Pushed override live")], outputs=[default_prompt, active_prompt, badge_holder, diff_text, agent_status])
        use_default_btn.click(fn=reset_to_default, inputs=agent_dropdown, outputs=[default_prompt, active_prompt, badge_holder, diff_text, agent_status])

        use_dml.change(fn=lambda enabled: gr.update(visible=enabled), inputs=use_dml, outputs=dml_top_k)

        run_btn.click(
            fn=stream_runner,
            inputs=[goal, scenario, fast, use_dml, dml_top_k],
            outputs=[
                metrics_card,
                timeline,
                planner_box,
                coder_box,
                reviewer_box,
                ops_box,
                aggregator_box,
                final_box,
                memory_all,
                memory_planner,
                memory_coder,
                memory_reviewer,
                memory_ops,
                memory_aggregator,
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
