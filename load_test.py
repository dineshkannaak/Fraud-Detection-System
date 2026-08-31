"""Small API load test without external load-testing infrastructure.

Usage:
    python load_test.py --url http://localhost:8000 --requests 100 --workers 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time

import requests


def run_one(url: str, index: int, api_key: str = "") -> tuple[float, int]:
    headers = {"X-API-Key": api_key} if api_key else {}
    payload = {"transaction_id": f"load-test-{index}", "features": [0.0] * 30}
    started = time.perf_counter()
    try:
        response = requests.post(f"{url.rstrip('/')}/predict", json=payload, headers=headers, timeout=10)
        return (time.perf_counter() - started) * 1000, response.status_code
    except requests.RequestException:
        return (time.perf_counter() - started) * 1000, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda i: run_one(args.url, i, args.api_key), range(args.requests)))
    elapsed = time.perf_counter() - started
    latencies = sorted(latency for latency, _ in results)
    successes = sum(200 <= status < 300 for _, status in results)
    errors = args.requests - successes

    def percentile(value: int) -> float:
        return latencies[min(len(latencies) - 1, int(len(latencies) * value / 100))]
    print(f"requests={args.requests} workers={args.workers} elapsed_s={elapsed:.2f}")
    print(f"successes={successes} errors={errors} error_rate={errors / args.requests:.2%}")
    print(f"throughput_rps={args.requests / elapsed:.2f} mean_ms={statistics.mean(latencies):.2f} p50_ms={percentile(50):.2f} p95_ms={percentile(95):.2f} p99_ms={percentile(99):.2f}")


if __name__ == "__main__":
    main()
