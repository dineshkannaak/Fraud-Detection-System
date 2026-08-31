from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from threading import local

import requests

ROOT = Path(__file__).resolve().parent
_thread_state = local()


def session() -> requests.Session:
    if not hasattr(_thread_state, "session"):
        adapter = requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=0)
        _thread_state.session = requests.Session()
        _thread_state.session.mount("http://", adapter)
    return _thread_state.session


def descendants(pid: int) -> set[int]:
    found = {pid}
    frontier = [pid]
    while frontier:
        current = frontier.pop()
        try:
            child_text = Path(f"/proc/{current}/task/{current}/children").read_text().strip()
            children = [int(value) for value in child_text.split()]
        except (FileNotFoundError, PermissionError, ValueError):
            children = []
        for child in children:
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def process_tree_rss_mb(pid: int) -> float:
    total_kb = 0
    for process_id in descendants(pid):
        try:
            for line in Path(f"/proc/{process_id}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kb += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    return total_kb / 1024


def wait_until_ready(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/health", timeout=1)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise RuntimeError("Stress server did not become ready")


def request_one(url: str, index: int, mode: str, api_key: str = "", batch_size: int = 100) -> tuple[float, int, int, str | None]:
    headers = {"X-API-Key": api_key} if api_key else {}
    if mode == "valid":
        payload = {"transaction_id": f"hard-valid-{index}", "features": [0.0] * 30}
        endpoint = "/predict"
    elif mode == "invalid":
        payload = {"transaction_id": f"hard-invalid-{index}", "features": [0.0] * 29}
        endpoint = "/predict"
    elif mode == "batch":
        one = {"transaction_id": f"hard-batch-{index}", "features": [0.0] * 30}
        payload = {"transactions": [one] * batch_size}
        endpoint = "/predict-batch"
    else:
        raise ValueError(mode)

    started = time.perf_counter()
    try:
        response = session().post(f"{url}{endpoint}", json=payload, headers=headers, timeout=30)
        return (time.perf_counter() - started) * 1000, response.status_code, 1, None
    except requests.RequestException as exc:
        return (time.perf_counter() - started) * 1000, 0, 0, type(exc).__name__


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def scenario(url: str, count: int, concurrency: int, mode: str, api_key: str = "", batch_size: int = 100) -> dict:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda index: request_one(url, index, mode, api_key, batch_size), range(count)))
    elapsed = time.perf_counter() - started
    latencies = [item[0] for item in results]
    statuses = [item[1] for item in results]
    status_counts = {str(status): statuses.count(status) for status in sorted(set(statuses))}
    successes = sum(200 <= status < 300 for status in statuses)
    return {
        "mode": mode,
        "requests": count,
        "transactions_requested": count * batch_size if mode == "batch" else count,
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_rps": round(count / elapsed, 2),
        "transaction_throughput": round((count * batch_size if mode == "batch" else count) / elapsed, 2),
        "successes": successes,
        "errors": count - successes,
        "error_rate": round((count - successes) / count, 6),
        "status_counts": status_counts,
        "latency_ms": {
            "min": round(min(latencies), 3),
            "mean": round(sum(latencies) / len(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3),
        },
    }


def run_server(url: str, port: int, workers: int, scale: int, batch_size: int, api_key: str = "") -> tuple[list[dict], float]:
    env = os.environ.copy()
    env.update({"RATE_LIMIT_PER_MINUTE": "1000000", "API_KEY": api_key, "ENABLE_SHAP": "false"})
    command = [sys.executable, "-m", "uvicorn", "stress_server:app", "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers)]
    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stop = threading.Event()
    rss_samples: list[float] = []

    def sample_memory() -> None:
        while not stop.is_set():
            rss_samples.append(process_tree_rss_mb(server.pid))
            time.sleep(0.05)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        wait_until_ready(url)
        rows = []
        # 1,000 requests at five concurrency levels is substantially heavier than
        # the original 200-request / four-level test while remaining reproducible.
        for mode, base_count, levels in [
            ("valid", 1000, [1, 10, 50, 100, 250]),
            ("invalid", 1000, [100]),
            ("batch", 100, [10, 50]),
        ]:
            count = base_count * scale
            for concurrency in levels:
                rows.append(scenario(url, count, concurrency, mode, api_key, batch_size))
        return rows, max(rss_samples, default=0.0)
    finally:
        stop.set()
        sampler.join(timeout=1)
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def run_rate_limit_and_auth(port: int) -> dict:
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update({"RATE_LIMIT_PER_MINUTE": "25", "API_KEY": "hard-secret", "ENABLE_SHAP": "false"})
    command = [sys.executable, "-m", "uvicorn", "stress_server:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"]
    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_until_ready(url)
        unauthenticated = requests.get(f"{url}/model-info", timeout=5).status_code
        authenticated = requests.get(f"{url}/model-info", headers={"X-API-Key": "hard-secret"}, timeout=5).status_code
        statuses = []
        for index in range(100):
            payload = {"transaction_id": f"rate-{index}", "features": [0.0] * 30}
            statuses.append(requests.post(f"{url}/predict", json=payload, headers={"X-API-Key": "hard-secret"}, timeout=10).status_code)
        return {
            "unauthenticated_protected_status": unauthenticated,
            "authenticated_protected_status": authenticated,
            "rate_limit_status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses))},
            "rate_limit_triggered": 429 in statuses,
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--workers", default="1,4,8")
    parser.add_argument("--output", default="outputs/hard_stress_before.json")
    parser.add_argument("--label", default="hard_stress")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    all_rows = []
    max_rss = 0.0
    for workers in [int(value) for value in args.workers.split(",")]:
        rows, rss = run_server(url, args.port, workers, args.scale, args.batch_size)
        for row in rows:
            row["workers"] = workers
        all_rows.extend(rows)
        max_rss = max(max_rss, rss)
    security = run_rate_limit_and_auth(args.port)
    report = {
        "test_name": args.label,
        "scale": args.scale,
        "requests_per_worker_set": int(sum(row["requests"] for row in all_rows) / len([int(value) for value in args.workers.split(",")])),
        "worker_counts": [int(value) for value in args.workers.split(",")],
        "total_http_requests": int(sum(row["requests"] for row in all_rows)),
        "total_transactions_requested": int(sum(row["transactions_requested"] for row in all_rows)),
        "max_process_tree_rss_mb": round(max_rss, 3),
        "scenarios": all_rows,
        "security_and_resilience": security,
        "limitations": [
            "The production model artifacts were not included, so deterministic mock inference was used.",
            "The stress run used local loopback HTTP rather than Render/cloud networking.",
            "SHAP was disabled for capacity measurement and must be tested separately if enabled.",
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
