from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def flatten(path: Path) -> pd.DataFrame:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scenario in report["scenarios"]:
        row = {key: value for key, value in scenario.items() if key not in {"status_counts", "latency_ms"}}
        row.update({f"latency_{key}_ms": value for key, value in scenario["latency_ms"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    path = ROOT / "outputs/hard_stress_20x_before.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    frame = flatten(path)
    valid = frame[frame["mode"] == "valid"]
    batch = frame[frame["mode"] == "batch"]
    invalid = frame[frame["mode"] == "invalid"]
    summary = {
        "test_name": report["test_name"],
        "total_http_requests": report["total_http_requests"],
        "total_transactions_requested": report["total_transactions_requested"],
        "max_process_tree_rss_mb": report["max_process_tree_rss_mb"],
        "valid_error_rate_max": float(valid["error_rate"].max()),
        "valid_worst_p95": valid.loc[valid["latency_p95_ms"].idxmax()].to_dict(),
        "valid_worst_p99": valid.loc[valid["latency_p99_ms"].idxmax()].to_dict(),
        "valid_best_throughput": valid.loc[valid["throughput_rps"].idxmax()].to_dict(),
        "batch_worst_p95": batch.loc[batch["latency_p95_ms"].idxmax()].to_dict(),
        "invalid_worst_p95": invalid.loc[invalid["latency_p95_ms"].idxmax()].to_dict(),
        "security": report["security_and_resilience"],
    }
    (ROOT / "outputs/hard_stress_20x_before_analysis.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    frame.to_csv(ROOT / "outputs/hard_stress_20x_before_summary.csv", index=False)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
