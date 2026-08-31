# Fraud Detection API Stress-Test Report

## Executive summary

The FastAPI request path was stress-tested over HTTP using the real application, middleware, validation, routes, response serialization, request IDs, counters, and metrics logic. The test used a deterministic in-memory model wrapper because the original project did not include trained model binaries. Across 1,600 valid prediction requests, the service returned 1,600 successful HTTP 200 responses with **zero errors**.

The highest measured throughput was **886.68 requests per second** with four Uvicorn workers and 20 concurrent clients. The highest measured P95 latency was **84.828 ms** and the highest P99 latency was **86.489 ms**, both at one worker and 50 concurrent clients. These results are below the project’s nominal 100 ms latency target in this local test, but they are not production-capacity guarantees because real model artifacts, cloud networking, TLS, proxies, and Render infrastructure were not included.

## Test setup

| Item | Configuration |
|---|---|
| Application | Actual `app.py` FastAPI application and middleware |
| HTTP server | Uvicorn on `127.0.0.1:8765` |
| Worker counts | 1 and 4 |
| Requests per scenario | 200 |
| Concurrency levels | 1, 5, 20, and 50 concurrent clients |
| Total requests | 1,600 |
| Payload | Valid 30-feature `[Time, V1..V28, Amount]` request |
| Model | Deterministic in-memory mock model; real production binaries were unavailable |
| SHAP | Disabled |
| Rate limit | Raised to 100,000 requests/minute for throughput measurement |
| Measurements | Throughput, status counts, error rate, min/mean/median/P95/P99/max latency, parent RSS sample |

The command used was:

```bash
python run_stress.py \
  --requests 200 \
  --concurrency 1,5,20,50 \
  --workers 1,4 \
  --port 8765 \
  --output outputs/stress_test_results.json
```

## Results by scenario

| Workers | Concurrency | Throughput (RPS) | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Max (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 413.17 | 2.341 | 2.261 | 2.774 | 2.996 | 5.829 | 0 |
| 1 | 5 | 563.34 | 8.326 | 7.978 | 12.669 | 14.263 | 15.075 | 0 |
| 1 | 20 | 567.82 | 33.076 | 33.181 | 42.738 | 44.559 | 46.608 | 0 |
| 1 | 50 | 546.49 | 77.394 | 80.623 | 84.828 | 86.489 | 86.727 | 0 |
| 4 | 1 | 333.18 | 2.920 | 2.285 | 3.217 | 36.930 | 38.520 | 0 |
| 4 | 5 | 815.75 | 5.518 | 5.210 | 8.919 | 10.257 | 10.540 | 0 |
| 4 | 20 | **886.68** | 17.810 | 17.427 | 31.503 | 34.790 | 35.190 | 0 |
| 4 | 50 | 797.26 | 18.866 | 16.683 | 42.549 | 51.500 | 66.466 | 0 |

## Interpretation

The one-worker service scaled from 413.17 RPS at a single client to a plateau near 550–568 RPS at 20–50 concurrent clients. Latency increased substantially as concurrency rose: P95 reached 84.828 ms at 50 clients. This indicates queueing pressure in a single process even though no requests failed.

The four-worker service achieved higher parallel throughput at moderate and high concurrency. Its best result occurred at 20 clients, where throughput reached 886.68 RPS and P95 latency was 31.503 ms. At 50 clients, throughput declined slightly to 797.26 RPS while P95 and P99 latency increased to 42.549 ms and 51.500 ms. This is consistent with approaching local CPU or scheduling limits rather than an application error condition.

All measured requests returned HTTP 200 and the application’s counters and response path remained operational. The sampled parent-process RSS maximum was 77.941 MB. Because Uvicorn workers are separate processes, this value must not be interpreted as total service memory; a production memory test must measure the complete process tree and the real model’s resident memory.

## Findings

| Finding | Assessment |
|---|---|
| Reliability under tested load | Strong: 1,600/1,600 successful requests; 0% measured error rate. |
| Best throughput | 886.68 RPS at four workers and 20 concurrent clients. |
| Latency target | P95 and P99 remained below 100 ms in every tested scenario. |
| Concurrency bottleneck | One worker showed sharp latency growth at 50 clients; four workers reduced the effect. |
| Rate limiter | Intentionally bypassed for capacity measurement; it must be tested separately. |
| Memory result | Parent RSS peaked at 77.941 MB; child-worker/model memory was not included. |
| Production confidence | Limited until rerun with real artifacts on the target deployment platform. |

## Limitations

The production XGBoost model and calibrated artifacts were not included in the supplied project, so the test exercised deterministic mock inference. It therefore measures the application and HTTP overhead more reliably than it measures real model inference cost. The run was performed in a local sandbox, not on Render or another cloud instance. It excluded TLS termination, reverse proxies, internet latency, autoscaling, container CPU/memory quotas, shared rate limiting, persistent metrics, and real SHAP computation.

The test used 200 requests per scenario, which is suitable for a smoke stress test but not a long-duration soak test. It did not test invalid-request floods, authentication failures, rate-limit responses, batch payloads, or large request bodies as performance scenarios.

## Recommended production acceptance test

Before declaring a live deployment ready, repeat this harness against the deployed API using the generated `fraud_model.pkl`, scalers, probability calibrator, and threshold. Run at least 5–15 minutes per load level, ramp through 1, 5, 20, 50, 100, and 250 concurrent clients, and record the complete process-tree memory. Repeat with SHAP enabled if it will be exposed to users. Run a separate security scenario with the configured API key and rate limiter, then verify that unauthorized requests return 401 and excess requests return 429 without destabilizing the service.

For a multi-replica deployment, use an external metrics backend and distributed rate limiter. Choose worker count based on the real model’s CPU and memory footprint rather than this mock-model result. Keep an error-rate threshold of 1% or lower and define an application-specific P95/P99 SLO before enabling automatic rollout.
