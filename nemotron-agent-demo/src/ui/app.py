from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List

import gradio as gr
import requests

from src.demo.orchestrator import run_demo_stream
from src.playground import cluster_manager
from src.playground import manager as playground_manager
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
DOCKER_PLAYGROUND_WARNING = (
    "<div class='banner warning'>Playground requires Docker CLI + /var/run/docker.sock mount.</div>"
)


def ping_server() -> bool:
    try:
        resp = requests.get(f"{OPENAI_BASE_URL}/models", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def docker_status_banner() -> str:
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except FileNotFoundError:
        return DOCKER_PLAYGROUND_WARNING
    except subprocess.TimeoutExpired:
        return DOCKER_PLAYGROUND_WARNING

    if result.returncode != 0:
        return DOCKER_PLAYGROUND_WARNING
    return ""


def badge(status: str) -> str:
    return f"<span class='badge {status}'>{status.title()}</span>"


def render_progress(stages: List[Dict]) -> str:
    done = len([s for s in stages if s["status"] == "done"])
    total = len(stages)
    pct = int((done / total) * 100) if total else 0
    return f"<div class='progress-shell'><div class='progress-fill' style='width:{pct}%;'></div></div>"


def render_dml_stream(dml_info: Dict) -> str:
    requested = bool(dml_info.get("requested"))
    enabled = bool(dml_info.get("enabled"))
    error = dml_info.get("error")
    counters = dml_info.get("counters", {}) or {}
    get_calls = counters.get("dml_get_calls_per_run", 0)
    ingest_calls = counters.get("dml_ingest_calls_per_run", 0)
    error_state = bool(error) and requested and not enabled

    request_state = "on" if requested else "off"
    cookbook_state = "on" if get_calls > 0 else "off"
    ingest_state = "on" if ingest_calls > 0 else "off"
    stream_state = "error" if error_state else ("on" if enabled else "off")
    error_note = f"<span class='dml-stream-error'>{error}</span>" if error_state else ""
    return (
        "<div class='dml-stream'>"
        "<div class='dml-stream-title'>DML stream</div>"
        f"<div class='dml-light {stream_state}' title='DML status'></div>"
        "<div class='dml-stream-steps'>"
        f"<div class='dml-step'><span class='dml-light {request_state}'></span>Requested</div>"
        f"<div class='dml-step'><span class='dml-light {cookbook_state}'></span>Cookbook</div>"
        f"<div class='dml-step'><span class='dml-light {ingest_state}'></span>Ingest</div>"
        f"{error_note}"
        "</div>"
        "</div>"
    )


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


def render_cookbook_panel(state: Dict) -> str:
    dml_info = state.get("dml", {})
    if not dml_info.get("requested"):
        return "<div class='card'><h3>Cookbook guidance (beginning of run)</h3><div>DML disabled.</div></div>"
    if not dml_info.get("enabled"):
        error = dml_info.get("error") or "DML unavailable"
        return (
            "<div class='card'><h3>Cookbook guidance (beginning of run)</h3>"
            f"<div class='memory-error'>DML error: {error}</div></div>"
        )
    cookbook = dml_info.get("cookbook", {})
    found = cookbook.get("found", False)
    sources = cookbook.get("sources", [])
    latency_ms = cookbook.get("latency_ms", 0)
    cookbook_text = cookbook.get("cookbook_text", "")
    status = "yes" if found else "no"
    source_text = ", ".join([str(source) for source in sources]) if sources else "None"
    details = "<div class='memory-empty'>No cookbook guidance found.</div>"
    if cookbook_text:
        details = (
            "<details>"
            "<summary>Cookbook text (collapsed)</summary>"
            f"<pre>{cookbook_text}</pre>"
            "</details>"
        )
    return (
        "<div class='card'>"
        "<h3>Cookbook guidance (beginning of run)</h3>"
        f"<div>Found: <strong>{status}</strong></div>"
        f"<div>Sources: {source_text}</div>"
        f"<div>Latency: {latency_ms} ms</div>"
        f"{details}"
        "</div>"
    )


def render_ingest_panel(state: Dict) -> str:
    dml_info = state.get("dml", {})
    if not dml_info.get("requested"):
        return "<div class='card'><h3>New run report ingested (end of run)</h3><div>DML disabled.</div></div>"
    if not dml_info.get("enabled"):
        error = dml_info.get("error") or "DML unavailable"
        return (
            "<div class='card'><h3>New run report ingested (end of run)</h3>"
            f"<div class='memory-error'>DML error: {error}</div></div>"
        )
    ingest = dml_info.get("ingest", {})
    if not ingest.get("ok"):
        error = ingest.get("error") or "Pending"
        return (
            "<div class='card'><h3>New run report ingested (end of run)</h3>"
            f"<div class='memory-empty'>Status: {error}</div></div>"
        )
    return (
        "<div class='card'>"
        "<h3>New run report ingested (end of run)</h3>"
        f"<div>Summary ID: {ingest.get('summary_id')}</div>"
        f"<div>Ingested ID: {ingest.get('ingested_id')}</div>"
        f"<div>Summary latency: {ingest.get('summary_latency_ms')} ms</div>"
        "</div>"
    )


def render_playground_status(state: Dict) -> str:
    playground = state.get("playground", {})
    if not playground.get("enabled"):
        return "<div class='card'><h3>Playground</h3><div>Playground disabled.</div></div>"
    status = str(playground.get("status") or "unknown").lower()
    if status == "running":
        badge_status = "running"
    elif status in {"removed", "exited"}:
        badge_status = "done"
    elif status in {"error", "missing"}:
        badge_status = "failed"
    else:
        badge_status = "queued"
    error = playground.get("error")
    warning = playground.get("warning")
    image = playground.get("image")
    requested_image = playground.get("requested_image")
    workspace_host = playground.get("workspace_host")
    workspace_container = playground.get("workspace_container")
    warning_html = f"<div class='banner warning'>{warning}</div>" if warning else ""
    error_html = f"<div class='memory-error'>Error: {error}</div>" if error else ""
    requested_html = ""
    if requested_image and requested_image != image:
        requested_html = f"<div>Requested image: {requested_image}</div>"
    workspace_html = ""
    if workspace_host or workspace_container:
        workspace_html = (
            "<div class='workspace-paths'>"
            f"<div>Host workspace: {workspace_host or '—'}</div>"
            f"<div>Container workspace: {workspace_container or '—'}</div>"
            "</div>"
        )
    return (
        "<div class='card'>"
        "<h3>Playground</h3>"
        f"{warning_html}"
        f"<div>{badge(badge_status)} Status: {status}</div>"
        f"<div>Container: {playground.get('name')}</div>"
        f"<div>Image: {image}</div>"
        f"{requested_html}"
        f"{workspace_html}"
        f"{error_html}"
        "</div>"
    )


def render_cluster_status(state: Dict) -> str:
    cluster = state.get("cluster", {})
    if not cluster.get("enabled"):
        return "<div class='card'><h3>Cluster</h3><div>Cluster disabled.</div></div>"
    status = str(cluster.get("status") or "unknown").lower()
    if status == "running":
        badge_status = "running"
    elif status in {"removed", "exited"}:
        badge_status = "done"
    elif status in {"error", "missing"}:
        badge_status = "failed"
    else:
        badge_status = "queued"
    error = cluster.get("error")
    error_html = f"<div class='memory-error'>Error: {error}</div>" if error else ""
    containers = cluster.get("containers", []) or []
    container_html = ", ".join([c.get("name", "") for c in containers]) if containers else "—"
    api_port = cluster.get("api_port")
    web_port = cluster.get("web_port")
    api_url = f"http://localhost:{api_port}" if api_port else "—"
    web_url = f"http://localhost:{web_port}" if web_port else "—"
    workspace_host = cluster.get("workspace_host")
    workspace_container = cluster.get("workspace_container")
    workspace_html = ""
    if workspace_host or workspace_container:
        workspace_html = (
            "<div class='workspace-paths'>"
            f"<div>Host workspace: {workspace_host or '—'}</div>"
            f"<div>Container workspace: {workspace_container or '—'}</div>"
            "</div>"
        )
    return (
        "<div class='card'>"
        "<h3>Cluster</h3>"
        f"<div>{badge(badge_status)} Status: {status}</div>"
        f"<div>Network: {cluster.get('network') or '—'}</div>"
        f"<div>Containers: {container_html}</div>"
        f"<div>API URL: {api_url}</div>"
        f"<div>Web URL: {web_url}</div>"
        f"{workspace_html}"
        f"{error_html}"
        "</div>"
    )


def render_cluster_validation(state: Dict) -> str:
    cluster = state.get("cluster", {})
    validation = cluster.get("validation", {})
    if not validation:
        return "No validation run yet."
    checks = validation.get("checks", [])
    lines = [f"Overall: {'PASS' if validation.get('ok') else 'FAIL'}"]
    for check in checks:
        status = "PASS" if check.get("ok") else "FAIL"
        detail = check.get("detail", "")
        lines.append(f"- {check.get('name')}: {status} {detail}")
    return "\n".join(lines)


def render_playground_log(state: Dict) -> str:
    def _truncate(text: str, limit: int = 2000) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...\n[truncated]"

    playground = state.get("playground", {})
    if not playground.get("enabled"):
        return "Playground disabled."
    log_entries = playground.get("log", []) or []
    if not log_entries:
        return "No playground commands executed yet."
    blocks = []
    for entry in log_entries:
        blocks.append(
            "\n".join(
                [
                    f"$ {entry.get('cmd')}",
                    f"Exit code: {entry.get('exit_code')}",
                    "Stdout:",
                    _truncate(entry.get("stdout", "") or ""),
                    "Stderr:",
                    _truncate(entry.get("stderr", "") or ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def _default_prompt_source() -> str:
    return "<div class='prompt-source'><span class='label'>Prompt source:</span> typed</div>"


def stream_runner(
    goal: str,
    scenario: str,
    fast: bool,
    use_dml: bool,
    dml_top_k: int,
    use_playground: bool,
    playground_image: str,
    auto_remove_playground: bool,
    use_cluster: bool,
    cluster_image: str,
    cluster_size: int,
    cluster_run_id: str,
):
    if not ping_server():
        raise gr.Error("Server not ready at /v1/models. Start docker compose first.")

    for state in run_demo_stream(
        goal,
        fast=fast,
        scenario=scenario or None,
        use_dml=use_dml,
        dml_top_k=dml_top_k,
        use_playground=use_playground,
        playground_image=playground_image,
        auto_remove_playground=auto_remove_playground,
        use_cluster=use_cluster,
        cluster_image=cluster_image,
        cluster_size=cluster_size,
        cluster_run_id=cluster_run_id or None,
    ):
        metrics_html = render_metrics(state)
        timeline_html = render_progress(state["stages"]) + render_dml_stream(state.get("dml", {})) + render_timeline(state)
        outputs = render_agent_outputs(state)
        final_text = state.get("final", "")
        cookbook_panel = render_cookbook_panel(state)
        ingest_panel = render_ingest_panel(state)
        playground_status = render_playground_status(state)
        playground_log = render_playground_log(state)
        cluster_status = render_cluster_status(state)
        cluster_validation = render_cluster_validation(state)
        playground_name = state.get("playground", {}).get("name", "")
        playground_workspace_host = state.get("playground", {}).get("workspace_host", "")
        playground_workspace_container = state.get("playground", {}).get("workspace_container", "")
        cluster_run_id = state.get("cluster", {}).get("run_id", "")
        cluster_workspace_host = state.get("cluster", {}).get("workspace_host", "")
        cluster_workspace_container = state.get("cluster", {}).get("workspace_container", "")
        ready_for_removal = bool(state.get("playground", {}).get("ready_for_removal"))
        status = str(state.get("playground", {}).get("status", "")).lower()
        remove_btn_update = gr.update(visible=ready_for_removal, interactive=status not in {"removed", "missing"})
        delete_btn_update = gr.update(visible=ready_for_removal and bool(playground_workspace_host))
        cluster_ready = bool(state.get("cluster", {}).get("ready_for_removal"))
        cluster_status_value = str(state.get("cluster", {}).get("status", "")).lower()
        destroy_btn_update = gr.update(visible=cluster_ready, interactive=cluster_status_value not in {"removed", "missing"})
        delete_cluster_btn_update = gr.update(visible=cluster_ready and bool(cluster_workspace_host))
        yield (
            metrics_html,
            timeline_html,
            cookbook_panel,
            ingest_panel,
            outputs.get("planner", ""),
            outputs.get("coder", ""),
            outputs.get("reviewer", ""),
            outputs.get("ops", ""),
            outputs.get("aggregator", ""),
            final_text,
            cookbook_panel,
            ingest_panel,
            playground_status,
            playground_name,
            playground_workspace_host,
            playground_workspace_container,
            playground_log,
            remove_btn_update,
            delete_btn_update,
            playground_name,
            cluster_status,
            cluster_run_id,
            cluster_workspace_host,
            cluster_workspace_container,
            cluster_validation,
            destroy_btn_update,
            delete_cluster_btn_update,
            cluster_run_id,
        )


def build_ui() -> gr.Blocks:
    css = CSS_PATH.read_text()
    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Nemoyron 3 Nano Agentic Playground") as demo:
        gr.Markdown("# Nemoyron 3 Nano Agentic Playground")
        docker_banner = gr.HTML()
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
                use_playground = gr.Checkbox(label="Use Playground Container", value=False)
                playground_image = gr.Textbox(label="Playground image", value="nemotron-playground:latest", visible=False)
                auto_remove_playground = gr.Checkbox(label="Auto-remove container after run", value=False, visible=False)
                use_cluster = gr.Checkbox(label="Use Cluster Playground", value=False)
                cluster_size = gr.Slider(label="Cluster size", minimum=3, maximum=5, step=1, value=4, visible=False)
                cluster_image = gr.Textbox(label="Cluster image", value="nemotron-playground:latest", visible=False)
                create_cluster_btn = gr.Button("Create Cluster", visible=False)
                validate_cluster_btn = gr.Button("Run Validation", visible=False)
                destroy_cluster_btn = gr.Button("Destroy Cluster", variant="stop", visible=False)
                run_btn = gr.Button("Run Demo", variant="primary")
                server_status = gr.HTML("", label="Server status")
                playground_status = gr.HTML("", label="Playground status")
                cluster_status = gr.HTML("", label="Cluster status")
                playground_name_display = gr.Textbox(label="Playground container", interactive=False)
                cluster_run_id_display = gr.Textbox(label="Cluster run ID", interactive=False)
                remove_playground_btn = gr.Button("Remove Playground Container", variant="stop", visible=False)
                delete_workspace_btn = gr.Button("Delete Workspace", variant="stop", visible=False)
                delete_cluster_workspace_btn = gr.Button("Delete Cluster Workspace", variant="stop", visible=False)
                remove_status = gr.Markdown(value="")
                delete_status = gr.Markdown(value="")
                delete_cluster_status = gr.Markdown(value="")
            with gr.Column(scale=2):
                cookbook_panel = gr.HTML(elem_classes=["card", "scroll-panel"])
                ingest_panel = gr.HTML(elem_classes=["card", "scroll-panel"])
                with gr.Tab("Timeline"):
                    metrics_card = gr.HTML()
                    timeline = gr.HTML(elem_classes=["card", "scroll-panel"])
                with gr.Tab("Agent Outputs"):
                    planner_box = gr.Textbox(label="Planner", lines=6, elem_classes=["text-panel"])
                    coder_box = gr.Textbox(label="Coder", lines=6, elem_classes=["text-panel"])
                    reviewer_box = gr.Textbox(label="Reviewer", lines=6, elem_classes=["text-panel"])
                    ops_box = gr.Textbox(label="Ops", lines=6, elem_classes=["text-panel"])
                    aggregator_box = gr.Textbox(label="Aggregator", lines=6, elem_classes=["text-panel"])
                with gr.Tab("Final Answer"):
                    final_box = gr.Textbox(label="Final", lines=8, elem_classes=["text-panel"])
                with gr.Tab("DML Cookbook"):
                    gr.Markdown("Cookbook guidance (beginning of run) and run report ingestion (end of run).")
                    dml_cookbook_tab = gr.HTML(elem_classes=["scroll-panel"])
                    dml_ingest_tab = gr.HTML(elem_classes=["scroll-panel"])
                with gr.Tab("Playground"):
                    with gr.Row():
                        playground_workspace_host = gr.Textbox(label="Host workspace path", interactive=False)
                        playground_workspace_container = gr.Textbox(label="Container workspace path", interactive=False)
                    gr.Markdown("Playground command execution log (truncated).")
                    playground_log = gr.Textbox(label="Playground log", lines=12, elem_classes=["text-panel"])
                with gr.Tab("Cluster"):
                    with gr.Row():
                        cluster_workspace_host = gr.Textbox(label="Host workspace path", interactive=False)
                        cluster_workspace_container = gr.Textbox(label="Container workspace path", interactive=False)
                    gr.Markdown("Cluster validation report.")
                    cluster_validation = gr.Textbox(label="Cluster validation", lines=10, elem_classes=["text-panel"])

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

        playground_name_state = gr.State("")
        cluster_run_id_state = gr.State("")

        def update_status():
            status = ping_server()
            pill_class = "up" if status else "down"
            text = "Online" if status else "Offline"
            return f"<div class='server-pill {pill_class}'>● Server {text}</div>"

        def toggle_playground(enabled: bool):
            return gr.update(visible=enabled), gr.update(visible=enabled)

        def toggle_cluster(enabled: bool):
            return (
                gr.update(visible=enabled),
                gr.update(visible=enabled),
                gr.update(visible=enabled),
                gr.update(visible=enabled),
                gr.update(visible=enabled),
            )

        def remove_playground_container(name: str):
            if not name:
                return (
                    "<div class='card'><h3>Playground</h3><div>No container available.</div></div>",
                    "No playground container to remove.",
                    gr.update(visible=True, interactive=False),
                )
            result = playground_manager.remove_playground(name)
            status = playground_manager.status(name)
            status_value = "removed" if result.get("ok") else status.get("status", "error")
            state = {
                "playground": {
                    "enabled": True,
                    "name": name,
                    "status": status_value,
                    "image": "",
                    "error": result.get("error"),
                }
            }
            message = "Playground container removed." if result.get("ok") else f"Remove failed: {result.get('error')}"
            return (
                render_playground_status(state),
                message,
                gr.update(visible=True, interactive=not result.get("ok")),
            )

        def remove_workspace(path: str):
            if not path:
                return "No workspace path available.", gr.update(visible=True, interactive=False)
            result = playground_manager.remove_workspace(path)
            message = "Workspace deleted." if result.get("ok") else f"Workspace delete failed: {result.get('error')}"
            return message, gr.update(visible=True, interactive=not result.get("ok"))

        def create_cluster(size: int, image: str):
            run_id = os.urandom(8).hex()
            cluster_state = cluster_manager.create_cluster(run_id, image, size, workspace_host=None)
            state = {"cluster": {"enabled": True, "run_id": run_id, **cluster_state}}
            status_html = render_cluster_status(state)
            validation_text = render_cluster_validation(state)
            destroy_update = gr.update(visible=True, interactive=cluster_state.get("ok", False))
            delete_update = gr.update(visible=bool(cluster_state.get("workspace_host")))
            return (
                status_html,
                run_id,
                cluster_state.get("workspace_host", ""),
                cluster_state.get("workspace_container", ""),
                validation_text,
                destroy_update,
                delete_update,
                run_id,
            )

        def validate_cluster(run_id: str):
            if not run_id:
                return "No cluster run ID available."
            validation = cluster_manager.validate_cluster(run_id)
            state = {"cluster": {"enabled": True, "run_id": run_id, "validation": validation}}
            return render_cluster_validation(state)

        def destroy_cluster(run_id: str):
            if not run_id:
                return (
                    "<div class='card'><h3>Cluster</h3><div>No cluster available.</div></div>",
                    "No cluster run ID available.",
                    gr.update(visible=True, interactive=False),
                )
            result = cluster_manager.destroy_cluster(run_id)
            status = "removed" if result.get("ok") else "error"
            state = {"cluster": {"enabled": True, "run_id": run_id, "status": status, "error": result.get("error")}}
            message = "Cluster destroyed." if result.get("ok") else f"Destroy failed: {result.get('error')}"
            return render_cluster_status(state), message, gr.update(visible=True, interactive=not result.get("ok"))

        demo.load(fn=docker_status_banner, inputs=None, outputs=docker_banner)
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
        use_playground.change(fn=toggle_playground, inputs=use_playground, outputs=[playground_image, auto_remove_playground])
        use_cluster.change(
            fn=toggle_cluster,
            inputs=use_cluster,
            outputs=[cluster_size, cluster_image, create_cluster_btn, validate_cluster_btn, destroy_cluster_btn],
        )

        run_btn.click(
            fn=stream_runner,
            inputs=[
                goal,
                scenario,
                fast,
                use_dml,
                dml_top_k,
                use_playground,
                playground_image,
                auto_remove_playground,
                use_cluster,
                cluster_image,
                cluster_size,
                cluster_run_id_state,
            ],
            outputs=[
                metrics_card,
                timeline,
                cookbook_panel,
                ingest_panel,
                planner_box,
                coder_box,
                reviewer_box,
                ops_box,
                aggregator_box,
                final_box,
                dml_cookbook_tab,
                dml_ingest_tab,
                playground_status,
                playground_name_display,
                playground_workspace_host,
                playground_workspace_container,
                playground_log,
                remove_playground_btn,
                delete_workspace_btn,
                playground_name_state,
                cluster_status,
                cluster_run_id_display,
                cluster_workspace_host,
                cluster_workspace_container,
                cluster_validation,
                destroy_cluster_btn,
                delete_cluster_workspace_btn,
                cluster_run_id_state,
            ],
            show_progress=True,
        )

        remove_playground_btn.click(
            fn=remove_playground_container,
            inputs=[playground_name_state],
            outputs=[playground_status, remove_status, remove_playground_btn],
        )

        delete_workspace_btn.click(
            fn=remove_workspace,
            inputs=[playground_workspace_host],
            outputs=[delete_status, delete_workspace_btn],
        )

        create_cluster_btn.click(
            fn=create_cluster,
            inputs=[cluster_size, cluster_image],
            outputs=[
                cluster_status,
                cluster_run_id_display,
                cluster_workspace_host,
                cluster_workspace_container,
                cluster_validation,
                destroy_cluster_btn,
                delete_cluster_workspace_btn,
                cluster_run_id_state,
            ],
        )

        validate_cluster_btn.click(
            fn=validate_cluster,
            inputs=[cluster_run_id_state],
            outputs=[cluster_validation],
        )

        destroy_cluster_btn.click(
            fn=destroy_cluster,
            inputs=[cluster_run_id_state],
            outputs=[cluster_status, delete_cluster_status, destroy_cluster_btn],
        )

        delete_cluster_workspace_btn.click(
            fn=remove_workspace,
            inputs=[cluster_workspace_host],
            outputs=[delete_cluster_status, delete_cluster_workspace_btn],
        )

    return demo


def main():
    demo = build_ui()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.queue()
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=port, share=False)


if __name__ == "__main__":
    main()
