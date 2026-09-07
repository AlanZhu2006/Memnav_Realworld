#!/usr/bin/env python3
"""Re-evaluate saved real-robot receipts without ROS, servers, or motion.

This is a post-hoc diagnostic, not a trajectory replay or a navigation test.
Only actually recorded evidence is evaluated; no synthetic cases are generated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.gpu.revisit_local_pose_adapter import decide_local_pose_handoff
from search_intent import forward_search_arc
from terminal_motion_override import terminal_motion_override


def read_rows(path):
    return [(number, json.loads(line)) for number, line in
            enumerate(path.read_text().splitlines(), 1) if line.strip()]


def seconds(row):
    return datetime.fromisoformat(row["received_utc"].replace("Z", "+00:00")).timestamp()


def numeric_summary(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return (None if not values else
            {"n": len(values), "median": statistics.median(values), "max": max(values)})


def audit_pair(root, pair):
    paths = [root / pair / name for name in
             ("cec_receipt.jsonl", "status.jsonl", "rgb_arrival_status.jsonl")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    active = [row for _, row in read_rows(paths[1]) if row["payload"].get("enabled") is True]
    if not active:
        raise ValueError(f"{pair}: no active interval")
    lower, upper = seconds(active[0]), seconds(active[-1])
    plans = [(line, row) for line, row in read_rows(paths[0])
             if lower <= seconds(row) <= upper and row["payload"].get("event") in
             ("imagegoal_plan", "imagegoal_plan_rejected_low_critic")]
    rows, decisions, default_mismatches = [], Counter(), []
    for line, row in plans:
        receipt = row["payload"]["receipt"]
        evidence = receipt.get("terminal_localization")
        arguments = dict(long_range_available=bool(receipt.get("cec_takeover")), evidence=evidence)
        unchanged = decide_local_pose_handoff(**arguments).audit_dict()
        # Exclude the schema version and new disclosure fields; compare the
        # old mode's actual decisions and commands to the saved deployment.
        fields = ("terminal_handoff_disposition", "terminal_controller_pointgoal_m",
                  "terminal_turn_error_left_rad", "terminal_stop_authorized")
        differences = [key for key in fields if receipt.get(key) != unchanged.get(key)]
        if differences:
            default_mismatches.append({"line": line, "fields": differences})
        updated = decide_local_pose_handoff(
            **arguments, metric_approach=True,
            expected_frame_index=receipt.get("cec_frame_idx")).audit_dict()
        combined = {**receipt, **updated}
        terminal = terminal_motion_override(combined, rotate_gain=1.5, max_angular_rps=0.55)
        low = bool(receipt.get("critic_fallback_applied"))
        decisions[updated["terminal_handoff_disposition"]] += 1
        d = receipt.get("navigation_diagnostics") or {}
        stamp = (d.get("input_timing") or {}).get("rgb_stamp_ns")
        rows.append({
            "source_line": line, "utc": row["received_utc"],
            "step": receipt.get("cec_step_index"), "frame": receipt.get("cec_frame_idx"),
            "anchor": receipt.get("cec_selected_anchor"),
            "low_critic": low,
            "old_disposition": receipt.get("terminal_handoff_disposition"),
            "new_disposition": updated["terminal_handoff_disposition"],
            "new_reason": updated["terminal_handoff_reason"],
            "predicted_distance_m": updated["terminal_predicted_distance_m"],
            "old_pointgoal_m": receipt.get("terminal_controller_pointgoal_m"),
            "new_pointgoal_m": updated["terminal_controller_pointgoal_m"],
            "new_turn_left_deg": (None if updated["terminal_turn_error_left_rad"] is None else
                                  math.degrees(updated["terminal_turn_error_left_rad"])),
            "metric_active": updated["terminal_metric_scale_control_authority"],
            "new_stop_authorized": updated["terminal_stop_authorized"],
            "low_critic_cannot_override_terminal": low and terminal.applied,
            "low_critic_default_hold": low and not terminal.applied,
            "source_to_receipt_s": None if not stamp else seconds(row) - stamp / 1e9,
        })
    # The final arrival publication precedes the disabled status. A 1 s
    # terminal tail includes it without counting it as an active plan.
    arrivals = [row for _, row in read_rows(paths[2]) if lower <= seconds(row) <= upper + 1.0]
    # Repeated published status does not constitute repeated independent matches.
    first_arrival = next((r["received_utc"] for r in arrivals
                          if r["payload"].get("arrival_latched") is True), None)
    report = {
        "pair": pair, "active_interval_utc": [active[0]["received_utc"], active[-1]["received_utc"]],
        "source_sha256": hashes, "active_plan_count": len(rows),
        "default_mode_command_mismatches": default_mismatches,
        "new_dispositions": dict(decisions),
        "metric_active_count": sum(r["metric_active"] for r in rows),
        "new_distance_only_stop_count": sum(r["new_stop_authorized"] for r in rows),
        "low_critic_count": sum(r["low_critic"] for r in rows),
        "low_critic_terminal_priority_count": sum(r["low_critic_cannot_override_terminal"] for r in rows),
        "old_source_to_receipt_s": numeric_summary(r["source_to_receipt_s"] for r in rows),
        "old_interplan_s": numeric_summary(seconds(b[1])-seconds(a[1]) for a, b in zip(plans, plans[1:])),
        "first_logged_visual_arrival_utc": first_arrival,
        "changed_terminal_rows": [r for r in rows if r["metric_active"] or r["low_critic"]],
    }
    report["source_files_unchanged"] = all(
        hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in paths)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", default=["pair_004", "pair_006", "pair_009"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    reports = [audit_pair(args.capture_root, pair) for pair in args.pairs]
    # Inspect the implementation's exact primitive, not a simulated robot.
    arc = forward_search_arc(-1)
    length = sum(math.dist(a[:2], b[:2]) for a, b in zip(arc, arc[1:]))
    result = {
        "schema": "real_recording_adapter_reassessment_v1_20260907",
        "scope": "recorded evidence re-evaluation only; not new SR or physical execution",
        "no_ros_no_servers_no_motion": True,
        "default_mode_matches_saved_commands": all(not r["default_mode_command_mismatches"] for r in reports),
        "source_files_unchanged": all(r["source_files_unchanged"] for r in reports),
        "no_distance_only_stop": all(r["new_distance_only_stop_count"] == 0 for r in reports),
        "optional_search_not_executed": {"polyline_length_m": length, "endpoint_xy": arc[-1, :2].tolist()},
        "pairs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
    for r in reports:
        print(json.dumps({k: v for k, v in r.items() if k != "changed_terminal_rows"}))
    if not (result["default_mode_matches_saved_commands"] and result["source_files_unchanged"]
            and result["no_distance_only_stop"]):
        raise SystemExit("recording reassessment found a discrepancy; inspect the saved report")


if __name__ == "__main__":
    main()
