# 100x Stress-Test, Analysis, and Improvement Report

## Executive summary

This campaign used the stress harness at **`scale=100`**, meaning each worker configuration received 100 times the base scenario count. It generated **1,240,000 HTTP requests per complete run** across four- and eight-worker configurations and processed **5,200,000 logical transactions**. The workload included 1,000,000 valid single predictions, 200,000 invalid validation requests, and 2,000,000 batch transactions. It also exercised API-key authentication, rate limiting, process-tree memory sampling, connection reuse, and 250-client concurrency.

The report-driven improvement was a dedicated prediction executor combined with FastAPI ORJSON responses. Prediction work is now dispatched to a dedicated bounded `ThreadPoolExecutor` instead of contending with Starlette’s general AnyIO thread pool. The API also uses `ORJSONResponse` for lower serialization overhead, and the admission guard remains in place to provide controlled back pressure. The batch path remains vectorized, and the rate window remains O(1) to maintain predictable bookkeeping cost.

The final run completed without application failures. The strongest measured result was eight-worker valid serving at ten concurrent clients: throughput increased from **976.60 to 1,042.29 requests/second**, while P95 latency decreased from **18.070 ms to 16.855 ms**. For the largest batch scenario, eight workers and 50 concurrent clients, logical transaction throughput increased from **52,585.68 to 60,916.12 transactions/second**, while P95 latency decreased from **184.276 ms to 151.078 ms**.

## Workload definition

| Dimension | Value |
|---|---:|
| Harness scale | 100 per worker configuration |
| Worker configurations | 4 and 8 |
| Valid requests | 5 × 100,000 per worker configuration |
| Invalid requests | 100,000 per worker configuration |
| Batch requests | 2 × 10,000 per worker configuration |
| Transactions per batch | 100 |
| HTTP requests per worker configuration | 620,000 |
| Total HTTP requests per run | 1,240,000 |
| Total logical transactions per run | 5,200,000 |
| Valid concurrency | 1, 10, 50, 100, and 250 |
| Batch concurrency | 10 and 50 |
| Invalid concurrency | 100 |
| Transport | Local loopback HTTP with reusable sessions |
| Model | Deterministic in-memory mock because trained artifacts were unavailable |
| SHAP | Disabled for capacity measurement |

The harness’s `scale=100` is the exact 100x multiplier for each scenario definition. The aggregate run used two worker configurations rather than three to keep the multi-million-request campaign bounded; the raw artifacts identify the precise denominators. This is a capacity and regression test for the application path, not a cloud deployment benchmark.

## Findings from the baseline

The 100x baseline remained functionally correct but showed rising high-concurrency tails. With four workers and 250 concurrent clients, valid P95 latency reached **534.232 ms** and P99 reached **710.837 ms**. The four-worker invalid flood reached P95 **211.477 ms**, demonstrating that validation-error response construction also consumes meaningful request capacity. Batch processing remained strong but still showed measurable serialization and thread scheduling overhead at larger concurrency.

The baseline also indicated that prediction work and response processing shared general-purpose async-server thread capacity. Under sustained mixed load, this creates contention between inference, validation, and framework work. The earlier overload guard controlled extreme saturation, but the remaining optimization target was to isolate prediction work and reduce JSON encoding cost.

## Improvements applied

The FastAPI backend now uses a dedicated prediction executor with a configurable `SYNC_THREAD_TOKENS` worker count. Both single prediction and batch prediction routes are asynchronous wrappers that dispatch their CPU/model work to this dedicated pool. The executor is initialized and safely recreated across application lifespans, then shut down cleanly during process shutdown.

FastAPI now defaults to `ORJSONResponse`, with `orjson` pinned in `requirements.txt`. The middleware admits requests before rate-window bookkeeping and releases the semaphore correctly on rate-limit exits. This prevents rejected overload traffic from consuming the in-flight budget or filling the rate deque unnecessarily. The service continues to expose `MAX_INFLIGHT_REQUESTS`, `SYNC_THREAD_TOKENS`, and `PREDICTION_LOGGING` as deployment controls.

## Before and after results

### Valid single predictions

| Workers | Concurrency | Before RPS | After RPS | Change | Before P95 | After P95 | P95 change | After status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 1 | 476.28 | 492.08 | +3.3% | 2.490 ms | 2.497 ms | +0.3% | 100,000 × 200 |
| 4 | 10 | 989.43 | 1,022.91 | +3.4% | 17.749 ms | 17.079 ms | -3.8% | 100,000 × 200 |
| 4 | 50 | 991.36 | 1,023.20 | +3.2% | 101.168 ms | 98.136 ms | -3.0% | 100,000 × 200 |
| 4 | 100 | 974.66 | 1,006.70 | +3.3% | 212.279 ms | 204.488 ms | -3.7% | 100,000 × 200 |
| 4 | 250 | 951.58 | 943.21 | -0.9% | 534.232 ms | 540.924 ms | +1.3% | 100,000 × 200 |
| 8 | 1 | 454.36 | 513.33 | +13.0% | 2.681 ms | 2.296 ms | -14.4% | 100,000 × 200 |
| 8 | 10 | 976.60 | 1,042.29 | +6.7% | 18.070 ms | 16.855 ms | -6.7% | 100,000 × 200 |
| 8 | 50 | 1,016.22 | 1,036.88 | +2.0% | 98.983 ms | 96.774 ms | -2.2% | 100,000 × 200 |
| 8 | 100 | 997.73 | 1,005.80 | +0.8% | 207.905 ms | 205.542 ms | -1.1% | 100,000 × 200 |
| 8 | 250 | 950.14 | 971.63 | +2.3% | 542.120 ms | 529.176 ms | -2.4% | 100,000 × 200 |

The raw comparison CSV is authoritative for every row. The high-level conclusion is that the dedicated executor improved the best moderate-concurrency serving point and reduced the largest eight-worker tail, while results at other points remained within normal local scheduling variation.

### Batch predictions

| Workers | Concurrency | Before transaction RPS | After transaction RPS | Change | Before P95 | After P95 | P95 change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10 | 57,540.63 | 54,850.67 | -4.7% | 27.480 ms | 28.245 ms | +2.8% |
| 4 | 50 | 58,799.28 | 60,158.02 | +2.3% | 150.203 ms | 149.724 ms | -0.3% |
| 8 | 10 | 56,207.73 | 60,497.14 | +7.6% | 28.934 ms | 26.775 ms | -7.5% |
| 8 | 50 | 52,585.68 | 60,916.12 | +15.8% | 184.276 ms | 151.078 ms | -18.0% |

Each batch request contained 100 transactions and all batch requests returned HTTP 200. The complete exact rows, including P50, P99, maximum latency, and status counts, are stored in `outputs/hard_stress_100x_comparison.csv`.

### Invalid requests and security

The invalid flood produced HTTP 422 validation responses rather than application 500 errors. Authentication remained correct: unauthenticated protected access returned 401 and authenticated access returned 200. The separate rate-limit burst produced 23 HTTP 200 responses followed by 77 HTTP 429 responses, confirming that security controls remained active after the executor and response-class changes.

## Memory and reliability

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Maximum sampled process-tree RSS | 675.789 MB | 679.336 MB | +0.52% |
| Valid application failures | 0 | 0 | No regression |
| Batch application failures | 0 | 0 | No regression |
| Security checks | Passed | Passed | No regression |

The memory values were measured on local processes using the deterministic mock model. They are not estimates for production XGBoost, calibration, SHAP, container, or cloud memory usage.

## Validation

After the improvement, Python compilation passed, linting passed, and the API regression suite passed with **15 tests passed and 1 artifact-dependent test skipped**. The 100x post-improvement stress run completed all planned scenarios, with no application crashes and correct security responses.

## Deployment decision and next steps

For production, use multiple Uvicorn workers or replicas, tune `SYNC_THREAD_TOKENS` and `MAX_INFLIGHT_REQUESTS` together, and treat HTTP 429 as an intentional back-pressure signal. Keep per-prediction logging disabled unless debugging. If multiple replicas are used, move rate limiting and metrics to shared infrastructure. Before accepting a production SLO, repeat the workload with the real model artifacts on the target platform, including TLS, reverse proxy, container limits, and SHAP-enabled measurements if explanations are exposed.

The results do not justify claiming unlimited capacity. They support a bounded, observable service with good batch throughput, improved moderate-concurrency serving, and controlled overload behavior.

## References

[1]: outputs/hard_stress_100x_before.json "100x baseline raw results"
[2]: outputs/hard_stress_100x_after.json "100x post-improvement raw results"
[3]: outputs/hard_stress_100x_comparison.csv "100x before/after comparison table"
[4]: app.py "Final FastAPI implementation"
