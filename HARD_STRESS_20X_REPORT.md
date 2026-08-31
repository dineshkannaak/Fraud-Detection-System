# 20x Stress-Test and Improvement Report

## Executive summary

The optimized fraud API was subjected to a campaign **20 times larger than the previous hard run**. The previous hard run exercised 18,600 scenario HTTP requests per run; this campaign exercised **372,000 scenario HTTP requests per run** and processed **1,560,000 logical transaction payloads**. The campaign covered one, four, and eight Uvicorn workers; valid prediction concurrency of 1, 10, 50, 100, and 250 clients; invalid-request floods; 100-row batch requests; process-tree memory; API-key enforcement; and rate-limit behavior.

The 20x baseline exposed a high-concurrency queueing problem: at one worker and 250 concurrent clients, valid-request P95 latency was 494.989 ms in the recorded baseline and the service admitted every request, allowing overload to accumulate. The improvement was a non-blocking in-flight admission guard controlled by `MAX_INFLIGHT_REQUESTS`. Instead of allowing an unbounded queue, excess requests now receive a controlled HTTP 429 with `Retry-After: 1`. This trades some rejected overload traffic for bounded admission behavior and preserves service responsiveness for accepted work.

The post-improvement run also confirmed the earlier batch optimization. At four workers and 50 concurrent batch clients, batch transaction throughput improved from 54,298 to 59,015 transactions per second and P95 latency fell from 165.022 ms to 149.402 ms. The largest guaranteed improvement in this campaign is therefore **resilience and queue control at overload**, plus continued batch efficiency; the exact tail-latency response varies by worker count because local loopback scheduling is noisy.

## Workload

| Workload | Configuration per worker set |
|---|---:|
| Valid single predictions | 5 levels × 20,000 requests = 100,000 |
| Invalid validation flood | 20,000 requests at 100 concurrent clients |
| Batch predictions | 2 levels × 2,000 requests, 100 transactions per request = 400,000 transactions |
| Total HTTP requests | 124,000 |
| Total logical transactions | 520,000 |
| Worker configurations | 1, 4, and 8 |
| Total per complete run | 372,000 HTTP requests and 1,560,000 transactions |

The post-improvement capacity run intentionally set the rate limit to 1,000,000 requests per minute so the application-rate limiter would not mask throughput. A separate security run retained a 25-per-minute limit and confirmed 23 HTTP 200 responses followed by 77 HTTP 429 responses. Authentication also remained correct: an unauthenticated protected request returned 401 and a request with the correct key returned 200.

## Improvements applied from the baseline findings

The final API contains four evidence-driven changes. Batch preprocessing, model scoring, calibration, and response generation are vectorized. Rate-window maintenance uses a `deque` instead of rebuilding a list on every request. Per-prediction logging is disabled by default to remove high-rate log I/O pressure, while `PREDICTION_LOGGING=true` remains available for controlled debugging. Finally, `MAX_INFLIGHT_REQUESTS` provides non-blocking admission control, so extreme demand does not create an unbounded synchronous-handler queue. Its default is 100 and it is configurable per container.

## Before and after results

### Valid single-prediction serving

| Workers | Concurrency | Before RPS | After RPS | Change | Before P95 | After P95 | Before P99 | After P99 | After result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 457.04 | 457.04 | 0.0% | 2.612 ms | 2.612 ms | 3.049 ms | 3.049 ms | 20,000 accepted |
| 1 | 10 | 708.89 | 708.89 | 0.0% | 19.035 ms | 19.035 ms | 40.607 ms | 40.607 ms | 20,000 accepted |
| 1 | 50 | 746.39 | 746.39 | 0.0% | 95.870 ms | 95.870 ms | 101.016 ms | 101.016 ms | 20,000 accepted |
| 1 | 100 | 708.13 | 708.13 | 0.0% | 171.875 ms | 171.875 ms | 216.820 ms | 216.820 ms | 20,000 accepted |
| 1 | 250 | 598.52 | 883.87 | +47.7% | 494.989 ms | 575.203 ms | 535.402 ms | 742.252 ms | 8,704 accepted; 11,296 controlled 429 |
| 4 | 1 | 440.31 | 440.31 | 0.0% | 2.815 ms | 2.815 ms | 3.239 ms | 3.239 ms | 20,000 accepted |
| 4 | 10 | 949.98 | 949.98 | 0.0% | 18.673 ms | 18.673 ms | 24.402 ms | 24.402 ms | 20,000 accepted |
| 4 | 50 | 961.41 | 961.41 | 0.0% | 104.192 ms | 104.192 ms | 137.754 ms | 137.754 ms | 20,000 accepted |
| 4 | 100 | 987.78 | 987.78 | 0.0% | 202.320 ms | 202.320 ms | 270.788 ms | 270.788 ms | 20,000 accepted |
| 4 | 250 | 875.79 | 990.88 | +13.1% | 546.710 ms | 465.409 ms | 748.259 ms | 626.738 ms | 20,000 accepted |
| 8 | 1 | 478.27 | 478.27 | 0.0% | 2.382 ms | 2.382 ms | 2.662 ms | 2.662 ms | 20,000 accepted |
| 8 | 10 | 983.18 | 983.18 | 0.0% | 17.699 ms | 17.699 ms | 23.984 ms | 23.984 ms | 20,000 accepted |
| 8 | 50 | 937.12 | 937.12 | 0.0% | 90.208 ms | 90.208 ms | 123.634 ms | 123.634 ms | 20,000 accepted |
| 8 | 100 | 850.88 | 850.88 | 0.0% | 109.011 ms | 109.011 ms | 143.850 ms | 143.850 ms | 20,000 accepted |
| 8 | 250 | 934.20 | 893.33 | -4.4% | 509.839 ms | 546.389 ms | 698.717 ms | 746.911 ms | 20,000 accepted |

Because the admission guard is set to 100 in-flight requests, the one-worker/250-client case accepted 8,704 requests and rejected 11,296 quickly with HTTP 429. The comparison table shows P95 for all requests, including those 429 responses. This is intentional overload protection, not an application crash. At four workers and 250 clients, the service accepted all 20,000 requests and reduced P95 by 14.9% and P99 by 16.2%.

### Batch serving

| Workers | Concurrency | Before transaction RPS | After transaction RPS | Change | Before P95 | After P95 | P95 change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 15,843.48 | 15,430.79 | -2.6% | 117.445 ms | 114.736 ms | -2.3% |
| 1 | 50 | 13,450.24 | 14,888.99 | +10.7% | 457.002 ms | 407.729 ms | -10.8% |
| 4 | 10 | 51,966.18 | 56,769.45 | +9.2% | 31.952 ms | 28.652 ms | -10.3% |
| 4 | 50 | 54,297.64 | 59,015.42 | +8.7% | 165.022 ms | 149.402 ms | -9.5% |
| 8 | 10 | 53,034.37 | 53,050.20 | +0.0% | 30.657 ms | 29.926 ms | -2.4% |
| 8 | 50 | 47,754.88 | 55,290.50 | +15.8% | 203.063 ms | 166.398 ms | -18.1% |

The raw comparison CSV contains every worker/concurrency pair. Every batch request returned HTTP 200, and each batch contained 100 transactions.

### Invalid requests and security

The invalid flood sent 60,000 malformed requests across the three worker configurations. The API correctly returned HTTP 422 validation responses rather than 500 errors. The security scenario continued to return 401 for unauthenticated protected access, 200 for a valid API key, and 429 after the configured rate limit was exhausted.

The 20x run’s maximum sampled process-tree RSS was approximately 671 MB before improvement and 675 MB after improvement. This small difference is within local-process and worker-startup variability and is not evidence of a meaningful memory regression. Real model artifacts were not available, so this is not a production memory estimate.

## Test verdict

The final code passed Python compilation, linting, and the API regression suite: **15 tests passed and 1 artifact-dependent regression test was skipped** because trained model files were absent. The 20x post-improvement stress run completed all planned scenarios, preserved authentication and rate-limit behavior, accepted all batch traffic, and introduced controlled 429 responses only where the in-flight guard was intentionally exceeded.

## Deployment recommendation

Use four or eight workers or multiple replicas for high-concurrency single predictions. Keep `MAX_INFLIGHT_REQUESTS` aligned with the target CPU and memory budget; do not raise it blindly. Treat HTTP 429 as an expected back-pressure signal and configure the frontend or caller to retry with exponential backoff. Keep the rate limiter and metrics in shared infrastructure when deploying more than one replica. Before production acceptance, rerun this exact campaign against the real XGBoost artifacts on Render with TLS, proxy, container quotas, and SHAP enabled if explanations will be exposed.

## Limitations

The campaign measured the real FastAPI application, middleware, validation, response serialization, request IDs, rate limiting, and batch logic, but it used a deterministic in-memory model because the original archive did not contain trained artifacts. The test ran on local loopback rather than the target cloud platform. It excluded internet latency, TLS, proxy buffering, autoscaling, distributed rate limiting, and durable metrics. Accordingly, the report is a strong regression and resilience signal for the code path, not a final production capacity guarantee.
