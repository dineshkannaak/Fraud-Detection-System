from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def flatten(path: Path) -> pd.DataFrame:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scenario in report["scenarios"]:
        row = {key: value for key, value in scenario.items() if key not in {"status_counts", "latency_ms"}}
        row.update({f"latency_{key}_ms": value for key, value in scenario["latency_ms"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    before = flatten(OUTPUTS / "hard_stress_before.json").add_suffix("_before")
    after = flatten(OUTPUTS / "hard_stress_after.json").add_suffix("_after")
    keys_before = ["mode_before", "workers_before", "concurrency_before"]
    keys_after = ["mode_after", "workers_after", "concurrency_after"]
    before = before.rename(columns=dict(zip(keys_before, ["mode", "workers", "concurrency"])))
    after = after.rename(columns=dict(zip(keys_after, ["mode", "workers", "concurrency"])))
    joined = before.merge(after, on=["mode", "workers", "concurrency"], how="outer", validate="one_to_one")
    joined["throughput_change_pct"] = (joined["throughput_rps_after"] / joined["throughput_rps_before"] - 1) * 100
    joined["p95_change_pct"] = (joined["latency_p95_ms_after"] / joined["latency_p95_ms_before"] - 1) * 100
    joined["p99_change_pct"] = (joined["latency_p99_ms_after"] / joined["latency_p99_ms_before"] - 1) * 100
    joined.to_csv(OUTPUTS / "hard_stress_comparison.csv", index=False)

    valid = joined[joined["mode"] == "valid"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for workers, group in valid.groupby("workers"):
        group = group.sort_values("concurrency")
        label = f"{workers} worker(s)"
        axes[0].plot(group["concurrency"], group["throughput_rps_before"], linestyle="--", marker="o", label=f"Before {label}")
        axes[0].plot(group["concurrency"], group["throughput_rps_after"], marker="o", label=f"After {label}")
        axes[1].plot(group["concurrency"], group["latency_p95_ms_before"], linestyle="--", marker="o", label=f"Before {label}")
        axes[1].plot(group["concurrency"], group["latency_p95_ms_after"], marker="o", label=f"After {label}")
    axes[0].set_title("Valid throughput: before vs after")
    axes[0].set_xlabel("Concurrent clients")
    axes[0].set_ylabel("Requests/second")
    axes[1].set_title("Valid P95 latency: before vs after")
    axes[1].set_xlabel("Concurrent clients")
    axes[1].set_ylabel("Milliseconds")
    for axis in axes:
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "hard_stress_comparison.png", dpi=160)
    plt.close(fig)

    def row_for(mode: str, workers: int, concurrency: int) -> dict:
        return joined[(joined["mode"] == mode) & (joined["workers"] == workers) & (joined["concurrency"] == concurrency)].iloc[0].to_dict()

    summary = {
        "before_total_http_requests": int(json.loads((OUTPUTS / "hard_stress_before.json").read_text())["total_http_requests"]),
        "after_total_http_requests": int(json.loads((OUTPUTS / "hard_stress_after.json").read_text())["total_http_requests"]),
        "before_max_process_tree_rss_mb": json.loads((OUTPUTS / "hard_stress_before.json").read_text())["max_process_tree_rss_mb"],
        "after_max_process_tree_rss_mb": json.loads((OUTPUTS / "hard_stress_after.json").read_text())["max_process_tree_rss_mb"],
        "best_after_valid_throughput": valid.loc[valid["throughput_rps_after"].idxmax()].to_dict(),
        "before_worst_valid_p95": before[before["mode"] == "valid"].loc[before[before["mode"] == "valid"]["latency_p95_ms_before"].idxmax()].to_dict(),
        "after_worst_valid_p95": after[after["mode"] == "valid"].loc[after[after["mode"] == "valid"]["latency_p95_ms_after"].idxmax()].to_dict(),
        "batch_4_workers_50_concurrency": row_for("batch", 4, 50),
        "all_after_valid_requests_successful": bool((valid["errors_after"] == 0).all()),
    }
    (OUTPUTS / "hard_stress_comparison_analysis.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
