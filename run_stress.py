from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent


def wait_until_ready(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=1).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise RuntimeError("Stress-test server did not become ready")


def process_rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def one_request(url: str, index: int) -> tuple[float, int, str | None]:
    payload = {"transaction_id": f"stress-{index}", "features": [0.0] * 30}
    started = time.perf_counter()
    try:
        response = requests.post(f"{url}/predict", json=payload, timeout=15)
        return (time.perf_counter() - started) * 1000, response.status_code, None
    except requests.RequestException as exc:
        return (time.perf_counter() - started) * 1000, 0, type(exc).__name__


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def run_scenario(url: str, total_requests: int, concurrency: int) -> dict:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda index: one_request(url, index), range(total_requests)))
    elapsed = time.perf_counter() - started
    latencies = [result[0] for result in results]
    statuses = [result[1] for result in results]
    successes = sum(200 <= status < 300 for status in statuses)
    status_counts = {str(status): statuses.count(status) for status in sorted(set(statuses))}
    return {
        "requests": total_requests,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_rps": round(total_requests / elapsed, 2),
        "successes": successes,
        "errors": total_requests - successes,
        "error_rate": round((total_requests - successes) / total_requests, 6),
        "status_counts": status_counts,
        "latency_ms": {
            "min": round(min(latencies), 3),
            "mean": round(statistics.mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", default="1,5,20,50")
    parser.add_argument("--workers", default="1,4")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", default="outputs/stress_test_results.json")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    scenarios = []
    max_rss = 0.0

    for workers in [int(value) for value in args.workers.split(",")]:
        env = os.environ.copy()
        env.update({"RATE_LIMIT_PER_MINUTE": "100000", "API_KEY": "", "ENABLE_SHAP": "false"})
        command = [sys.executable, "-m", "uvicorn", "stress_server:app", "--host", "127.0.0.1", "--port", str(args.port), "--workers", str(workers)]
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        sampler_stop = threading.Event()
        rss_samples: list[float] = []

        def sample_memory() -> None:
            while not sampler_stop.is_set():
                value = process_rss_mb(server.pid)
                if value is not None:
                    rss_samples.append(value)
                time.sleep(0.05)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        try:
            wait_until_ready(url)
            for concurrency in [int(value) for value in args.concurrency.split(",")]:
                result = run_scenario(url, args.requests, concurrency)
                result["workers"] = workers
                scenarios.append(result)
        finally:
            sampler_stop.set()
            sampler.join(timeout=1)
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        max_rss = max(max_rss, max(rss_samples, default=0.0))

    report = {
        "test_type": "HTTP stress test against real FastAPI routes and middleware with deterministic mock artifacts",
        "server": "uvicorn stress_server:app",
        "requests_per_scenario": args.requests,
        "concurrency_levels": [int(value) for value in args.concurrency.split(",")],
        "worker_counts": [int(value) for value in args.workers.split(",")],
        "max_parent_process_rss_mb": round(max_rss, 3),
        "scenarios": scenarios,
        "limitations": [
            "Production model binaries were not included, so deterministic mock inference was used.",
            "The test ran on the local sandbox rather than the target Render/cloud instance.",
            "Network, TLS termination, proxy, database, and shared-rate-limiter effects were not included.",
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
