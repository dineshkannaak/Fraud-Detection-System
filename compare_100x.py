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
    before = flatten(OUTPUTS / "hard_stress_100x_before.json", "_before")
    after = flatten(OUTPUTS / "hard_stress_100x_after.json", "_after")
    key_before = ["mode_before", "workers_before", "concurrency_before"]
    key_after = ["mode_after", "workers_after", "concurrency_after"]
    before = before.rename(columns=dict(zip(key_before, ["mode", "workers", "concurrency"])))
    after = after.rename(columns=dict(zip(key_after, ["mode", "workers", "concurrency"])))
    joined = before.merge(after, on=["mode", "workers", "concurrency"], how="outer", validate="one_to_one")
    joined["throughput_change_pct"] = (joined["throughput_rps_after"] / joined["throughput_rps_before"] - 1) * 100
    joined["p95_change_pct"] = (joined["latency_p95_ms_after"] / joined["latency_p95_ms_before"] - 1) * 100
    joined["p99_change_pct"] = (joined["latency_p99_ms_after"] / joined["latency_p99_ms_before"] - 1) * 100
    joined.to_csv(OUTPUTS / "hard_stress_100x_comparison.csv", index=False)

    valid = joined[joined["mode"] == "valid"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for workers, group in valid.groupby("workers"):
        group = group.sort_values("concurrency")
        label = f"{workers} worker(s)"
        axes[0].plot(group["concurrency"], group["latency_p95_ms_before"], linestyle="--", marker="o", label=f"Before {label}")
        axes[0].plot(group["concurrency"], group["latency_p95_ms_after"], marker="o", label=f"After {label}")
        axes[1].plot(group["concurrency"], group["throughput_rps_before"], linestyle="--", marker="o", label=f"Before {label}")
        axes[1].plot(group["concurrency"], group["throughput_rps_after"], marker="o", label=f"After {label}")
    axes[0].set_title("100x valid P95 latency")
    axes[0].set_xlabel("Concurrent clients")
    axes[0].set_ylabel("Milliseconds")
    axes[1].set_title("100x valid throughput")
    axes[1].set_xlabel("Concurrent clients")
    axes[1].set_ylabel("Requests/second")
    for axis in axes:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUTS / "hard_stress_100x_comparison.png", dpi=160)
    plt.close(figure)

    before_report = json.loads((OUTPUTS / "hard_stress_100x_before.json").read_text(encoding="utf-8"))
    after_report = json.loads((OUTPUTS / "hard_stress_100x_after.json").read_text(encoding="utf-8"))
    batch = joined[joined["mode"] == "batch"]
    summary = {
        "before_total_http_requests": before_report["total_http_requests"],
        "after_total_http_requests": after_report["total_http_requests"],
        "before_total_transactions_requested": before_report["total_transactions_requested"],
        "after_total_transactions_requested": after_report["total_transactions_requested"],
        "before_max_process_tree_rss_mb": before_report["max_process_tree_rss_mb"],
        "after_max_process_tree_rss_mb": after_report["max_process_tree_rss_mb"],
        "valid_before_accepted": int(valid["successes_before"].sum()),
        "valid_after_accepted": int(valid["successes_after"].sum()),
        "valid_after_rejected": int(valid["errors_after"].sum()),
        "valid_after_rejection_rate": float(valid["errors_after"].sum() / valid["requests_after"].sum()),
        "best_after_valid_throughput": valid.loc[valid["throughput_rps_after"].idxmax()].to_dict(),
        "worst_before_valid_p95": valid.loc[valid["latency_p95_ms_before"].idxmax()].to_dict(),
        "worst_after_valid_p95": valid.loc[valid["latency_p95_ms_after"].idxmax()].to_dict(),
        "best_after_batch_transaction_throughput": batch.loc[batch["transaction_throughput_after"].idxmax()].to_dict(),
        "worst_after_batch_p95": batch.loc[batch["latency_p95_ms_after"].idxmax()].to_dict(),
        "after_security": after_report["security_and_resilience"],
    }
    (OUTPUTS / "hard_stress_100x_comparison_analysis.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
