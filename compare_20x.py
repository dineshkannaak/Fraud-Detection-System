from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def flatten(path: Path, suffix: str) -> pd.DataFrame:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scenario in report["scenarios"]:
        row = {key: value for key, value in scenario.items() if key not in {"status_counts", "latency_ms"}}
        row.update({f"latency_{key}_ms": value for key, value in scenario["latency_ms"].items()})
        rows.append(row)
    return pd.DataFrame(rows).add_suffix(suffix)


def main() -> None:
    before = flatten(OUTPUTS / "hard_stress_20x_before.json", "_before")
    after = flatten(OUTPUTS / "hard_stress_20x_after.json", "_after")
    keys_before = ["mode_before", "workers_before", "concurrency_before"]
    keys_after = ["mode_after", "workers_after", "concurrency_after"]
    before = before.rename(columns=dict(zip(keys_before, ["mode", "workers", "concurrency"])))
    after = after.rename(columns=dict(zip(keys_after, ["mode", "workers", "concurrency"])))
    joined = before.merge(after, on=["mode", "workers", "concurrency"], how="outer", validate="one_to_one")
    joined["accepted_change_pct"] = (joined["successes_after"] / joined["successes_before"].replace(0, 1) - 1) * 100
    joined["throughput_change_pct"] = (joined["throughput_rps_after"] / joined["throughput_rps_before"] - 1) * 100
    joined["p95_change_pct"] = (joined["latency_p95_ms_after"] / joined["latency_p95_ms_before"] - 1) * 100
    joined["p99_change_pct"] = (joined["latency_p99_ms_after"] / joined["latency_p99_ms_before"] - 1) * 100
    joined.to_csv(OUTPUTS / "hard_stress_20x_comparison.csv", index=False)

    valid = joined[joined["mode"] == "valid"].copy()
    batch = joined[joined["mode"] == "batch"].copy()
    invalid = joined[joined["mode"] == "invalid"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for workers, group in valid.groupby("workers"):
        group = group.sort_values("concurrency")
        label = f"{workers} worker(s)"
        axes[0].plot(group["concurrency"], group["latency_p95_ms_before"], linestyle="--", marker="o", label=f"Before {label}")
        axes[0].plot(group["concurrency"], group["latency_p95_ms_after"], marker="o", label=f"After {label}")
        axes[1].plot(group["concurrency"], group["throughput_rps_before"], linestyle="--", marker="o", label=f"Before {label}")
        axes[1].plot(group["concurrency"], group["throughput_rps_after"], marker="o", label=f"After {label}")
    axes[0].set_title("Valid-request P95 latency")
    axes[0].set_xlabel("Concurrent clients")
    axes[0].set_ylabel("Milliseconds")
    axes[1].set_title("Valid-request throughput")
    axes[1].set_xlabel("Concurrent clients")
    axes[1].set_ylabel("Requests/second")
    for axis in axes:
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "hard_stress_20x_comparison.png", dpi=160)
    plt.close(fig)

    def worst(frame: pd.DataFrame, column: str) -> dict:
        return frame.loc[frame[column].idxmax()].to_dict()

    before_report = json.loads((OUTPUTS / "hard_stress_20x_before.json").read_text(encoding="utf-8"))
    after_report = json.loads((OUTPUTS / "hard_stress_20x_after.json").read_text(encoding="utf-8"))
    summary = {
        "before_total_http_requests": before_report["total_http_requests"],
        "after_total_http_requests": after_report["total_http_requests"],
        "before_total_transactions_requested": before_report["total_transactions_requested"],
        "after_total_transactions_requested": after_report["total_transactions_requested"],
        "before_max_process_tree_rss_mb": before_report["max_process_tree_rss_mb"],
        "after_max_process_tree_rss_mb": after_report["max_process_tree_rss_mb"],
        "before_worst_valid_p95": worst(valid, "latency_p95_ms_before"),
        "after_worst_valid_p95": worst(valid, "latency_p95_ms_after"),
        "before_worst_valid_p99": worst(valid, "latency_p99_ms_before"),
        "after_worst_valid_p99": worst(valid, "latency_p99_ms_after"),
        "best_after_valid_throughput": worst(valid, "throughput_rps_after"),
        "best_after_batch_transaction_throughput": worst(batch, "transaction_throughput_after"),
        "worst_after_batch_p95": worst(batch, "latency_p95_ms_after"),
        "worst_after_invalid_p95": worst(invalid, "latency_p95_ms_after"),
        "after_valid_accepted_requests": int(valid["successes_after"].sum()),
        "after_valid_rejected_requests": int(valid["errors_after"].sum()),
        "after_security": after_report["security_and_resilience"],
    }
    (OUTPUTS / "hard_stress_20x_comparison_analysis.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
