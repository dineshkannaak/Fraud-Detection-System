from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "stress_test_results.json"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = []
    for scenario in report["scenarios"]:
        rows.append(
            {
                "workers": scenario["workers"],
                "concurrency": scenario["concurrency"],
                "requests": scenario["requests"],
                "throughput_rps": scenario["throughput_rps"],
                "successes": scenario["successes"],
                "errors": scenario["errors"],
                "error_rate": scenario["error_rate"],
                **{f"latency_{key}_ms": value for key, value in scenario["latency_ms"].items()},
            }
        )
    frame = pd.DataFrame(rows).sort_values(["workers", "concurrency"])
    frame.to_csv(OUTPUT_DIR / "stress_test_summary.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for workers, group in frame.groupby("workers"):
        axes[0].plot(group["concurrency"], group["throughput_rps"], marker="o", label=f"{workers} worker(s)")
        axes[1].plot(group["concurrency"], group["latency_p95_ms"], marker="o", label=f"{workers} worker(s)")
    axes[0].set_title("Throughput vs concurrency")
    axes[0].set_xlabel("Concurrent clients")
    axes[0].set_ylabel("Requests per second")
    axes[1].set_title("P95 latency vs concurrency")
    axes[1].set_xlabel("Concurrent clients")
    axes[1].set_ylabel("Milliseconds")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "stress_test_performance.png", dpi=160)
    plt.close(figure)

    summary = {
        "peak_throughput_rps": float(frame["throughput_rps"].max()),
        "peak_throughput_scenario": frame.loc[frame["throughput_rps"].idxmax()].to_dict(),
        "max_p95_latency_ms": float(frame["latency_p95_ms"].max()),
        "max_p99_latency_ms": float(frame["latency_p99_ms"].max()),
        "max_error_rate": float(frame["error_rate"].max()),
        "all_requests_successful": bool((frame["errors"] == 0).all()),
    }
    (OUTPUT_DIR / "stress_test_analysis.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
