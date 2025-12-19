"""Isaac-style robotics logging demo using the DML memory service."""
from __future__ import annotations

import sys

from app.isaac_dml_logger import log_event, query_robot_history


def main() -> None:
    log_event(
        "isaac_team1",
        "robot_arm_01",
        "mission_001",
        "Aborted pick at 14:03 due to repeated grasp failures in bin 3.",
        meta={"phase": "execution", "outcome": "failure"},
    )
    log_event(
        "isaac_team1",
        "robot_arm_01",
        "mission_001",
        "Observed glare in overhead camera, depth noise high.",
        meta={"phase": "diagnostics"},
    )
    log_event(
        "isaac_team1",
        "robot_arm_01",
        "mission_002",
        "Successful pick sequence after adjusting grasp approach angle.",
        meta={"phase": "execution", "outcome": "success"},
    )
    history = query_robot_history(
        tenant_id="isaac_team1",
        robot_id="robot_arm_01",
        query="Why is robot_arm_01 avoiding bin 3?",
    )
    print("=== Retrieved history for robot_arm_01 ===")
    print(history.get("raw_context", "No context"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - demo utility
        sys.stderr.write(f"isaac_demo failed: {exc}\n")
        sys.exit(1)
