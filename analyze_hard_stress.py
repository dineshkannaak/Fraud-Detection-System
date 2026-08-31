from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


def flatten(path: Path) -> pd.DataFrame:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scenario in report["scenarios"]:
        row = {key: value for key, value in scenario.items() if key != "latency_ms"}
        row.update({f"latency_{key}_ms": value for key, value in scenario["latency_ms"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    before = flatten(ROOT / "outputs/hard_stress_before.json")
    before.to_csv(ROOT / "outputs/hard_stress_before_summary.csv", index=False)
    valid = before[before["mode"] == "valid"]
    summary = {
        "total_http_requests": int(before["requests"].sum()),
        "valid_requests": int(valid["requests"].sum()),
        "valid_error_rate_max": float(valid["error_rate"].max()),
        "worst_valid_p95": valid.loc[valid["latency_p95_ms"].idxmax()].to_dict(),
        "worst_valid_p99": valid.loc[valid["latency_p99_ms"].idxmax()].to_dict(),
        "best_valid_throughput": valid.loc[valid["throughput_rps"].idxmax()].to_dict(),
        "worst_batch_p95": before[before["mode"] == "batch"].loc[before[before["mode"] == "batch"]["latency_p95_ms"].idxmax()].to_dict(),
        "invalid_validation_p95_max": float(before[before["mode"] == "invalid"]["latency_p95_ms"].max()),
        "max_memory_mb": float(json.loads((ROOT / "outputs/hard_stress_before.json").read_text())["max_process_tree_rss_mb"]),
    }
    (ROOT / "outputs/hard_stress_before_analysis.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for workers, group in valid.groupby("workers"):
        axes[0].plot(group["concurrency"], group["throughput_rps"], marker="o", label=f"{workers} worker(s)")
        axes[1].plot(group["concurrency"], group["latency_p95_ms"], marker="o", label=f"{workers} worker(s)")
    axes[0].set_title("Baseline throughput")
    axes[0].set_xlabel("Concurrent clients")
    axes[0].set_ylabel("Requests/second")
    axes[1].set_title("Baseline valid-request P95")
    axes[1].set_xlabel("Concurrent clients")
    axes[1].set_ylabel("Milliseconds")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/hard_stress_before.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
