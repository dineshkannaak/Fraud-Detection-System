# Hard Stress-Test Improvement Report

## Executive conclusion

The fraud API was tested with a workload more than ten times heavier than the previous stress run. The earlier test used 1,600 scenario requests; this campaign used **18,600 scenario requests per run**, plus authentication and rate-limit resilience checks. The service was then improved from the measured bottlenecks and the identical campaign was rerun.

The most important improvement was vectorized batch inference. At four workers and 50 concurrent batch clients, transaction throughput increased from **14,239 to 50,711 transactions per second**, a **256.1% increase**, while P95 latency fell from **432.192 ms to 65.950 ms**, an **84.7% reduction**. The same scenario’s P99 latency fell by **81.8%**.

Single-transaction serving remained reliable, but the one-worker 250-concurrent-client case remained a saturation point. Its P95 latency changed from 1,596.606 ms to 1,867.754 ms in this run. The practical deployment conclusion is to use multiple workers or replicas for high concurrency and to repeat the campaign with the real model artifacts before setting a production SLO.

## Workload and environment

| Dimension | Configuration |
|---|---|
| HTTP scenarios | Valid single prediction, invalid request flood, and 100-row batch prediction |
| Requests per worker configuration | 6,200 scenario requests: 5,000 valid, 1,000 invalid, and 200 batch HTTP requests |
| Worker configurations | 1, 4, and 8 Uvicorn workers |
| Valid concurrency levels | 1, 10, 50, 100, and 250 |
| Batch concurrency levels | 10 and 50 |
| Total scenario requests per run | 18,600 |
| Comparison | Same workload before and after code changes |
| Model artifacts | Deterministic mock model because trained production binaries were absent |
| Transport | Local loopback HTTP |
| SHAP | Disabled during capacity testing |
| Rate limit in capacity scenarios | Raised to 1,000,000/minute to avoid masking capacity |
| Security scenario | API key rejection/acceptance and 100-request rate-limit burst |

The security checks passed in both runs: an unauthenticated protected request returned 401, an authenticated request returned 200, and a 25-per-minute limit produced 23 HTTP 200 responses followed by 77 HTTP 429 responses.

## Baseline findings

The original hard run identified three practical bottlenecks. First, the original batch route called the single-transaction scoring function once for every row, repeating scaling, model invocation, calibration, response construction, and logging. This produced P95 batch latency of 432.192 ms at four workers and 50 concurrent clients. Second, the request middleware rebuilt a list of all recent rate-limit timestamps for every request, creating unnecessary work as the window grew. Third, every prediction emitted an INFO log, adding synchronization and I/O pressure under a high request rate. The one-worker 250-client scenario also showed severe queueing, with P95 latency of 1,596.606 ms.

## Implemented improvements

The API now preprocesses a batch with one NumPy operation, calls the model once for the complete batch, calibrates all probabilities together, and constructs typed responses from the resulting vector. The rate-limit window uses a `deque` and removes expired timestamps from the left in constant amortized time. Per-prediction logging is disabled by default and can be enabled explicitly with `PREDICTION_LOGGING=true`. Sync-thread capacity is configurable through `SYNC_THREAD_TOKENS` and defaults to 100, allowing deployment-specific tuning. The stress wrapper’s fake model was also corrected to support vectorized batch output, and the harness now samples the complete Uvicorn process tree’s RSS rather than only the parent process.

## Key before/after comparison

| Scenario | Before throughput | After throughput | Throughput change | Before P95 | After P95 | P95 change |
|---|---:|---:|---:|---:|---:|---:|
| Valid, 4 workers, 50 clients | 1,004.53 RPS | 1,014.32 RPS | +1.0% | 83.446 ms | 82.281 ms | -1.4% |
| Valid, 4 workers, 250 clients | 745.76 RPS | 827.15 RPS | +10.9% | 49.812 ms | 51.551 ms | +3.5% |
| Valid, 8 workers, 10 clients | 880.45 RPS | 983.18 RPS | +11.7% | 20.121 ms | 17.699 ms | -12.0% |
| Valid, 8 workers, 100 clients | 861.59 RPS | 850.88 RPS | -1.2% | 116.752 ms | 109.011 ms | -6.6% |
| Valid, 1 worker, 250 clients | 476.19 RPS | 386.69 RPS | -18.8% | 1,596.606 ms | 1,867.754 ms | +17.0% |
| Invalid, 4 workers, 100 clients | 853.05 RPS | 976.21 RPS | +14.4% | 88.012 ms | 79.358 ms | -9.8% |
| Batch, 4 workers, 50 clients | 14,239 tx/s | 50,711 tx/s | **+256.1%** | 432.192 ms | **65.950 ms** | **-84.7%** |
| Batch, 8 workers, 50 clients | 31,655 tx/s | 44,140 tx/s | +39.4% | 90.898 ms | 54.701 ms | -39.8% |

All valid and batch requests completed successfully in both runs. Invalid requests correctly returned HTTP 422 and were counted as expected validation outcomes rather than server failures.

## Resource behavior

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Maximum sampled process-tree RSS | 669.918 MB | 668.375 MB | -0.23% |
| Valid-request application errors | 0 | 0 | No regression |
| Batch-request application errors | 0 | 0 | No regression |
| Security/rate-limit behavior | Passed | Passed | No regression |

The memory number is the maximum sampled RSS of the local Uvicorn process tree using the deterministic mock model. It is not a production memory estimate for XGBoost, SHAP, or a cloud container with real artifacts.

## Interpretation and deployment decision

The batch path is materially better and is suitable for workloads that can group transactions. The single-prediction path is stable at moderate concurrency with four or eight workers, but high concurrency against one worker is not a suitable production configuration. The best measured valid single-request point in the final run was approximately 1,014 RPS at four workers and 50 concurrent clients, although repeated runs show normal local scheduling variance.

The code should be deployed with multiple workers or replicas, but worker count must be selected using the real model’s CPU and memory footprint. The default `SYNC_THREAD_TOKENS=100` should be tuned against the target container size. If request latency is more important than accepting every overloaded request, the next production hardening step should add explicit queue limits and load shedding so excess demand returns 429 quickly instead of waiting in an unbounded queue.

## Validation after the improvement

The final code passed compilation, linting, and the API test suite: **15 tests passed and 1 artifact-dependent regression test was skipped** because trained artifacts were not supplied. The hard-stress campaign also passed its security and rate-limit checks, and the final run generated raw JSON, comparison CSV, aggregate analysis JSON, and before/after performance charts under `outputs/`.

## Limitations and next test

The test used mock inference because `fraud_model.pkl`, scalers, calibrator, and threshold artifacts were not in the original project. It also used local loopback HTTP rather than Render, so it excludes TLS, reverse proxies, cloud CPU quotas, internet latency, autoscaling, and shared rate limiting. SHAP was disabled. Before production acceptance, rerun the same harness for 5–15 minutes per load level using the real artifacts, include a 100/250/500-client ramp, test SHAP separately, measure complete container memory, and confirm the chosen P95/P99 SLO on the deployed platform.
