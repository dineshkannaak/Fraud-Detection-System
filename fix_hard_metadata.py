from __future__ import annotations

import json
from pathlib import Path

for filename in ["hard_stress_before.json", "hard_stress_after.json"]:
    path = Path(__file__).resolve().parent / "outputs" / filename
    report = json.loads(path.read_text(encoding="utf-8"))
    report["total_http_requests"] = int(sum(row["requests"] for row in report["scenarios"]))
    report["total_transactions_requested"] = int(sum(row["transactions_requested"] for row in report["scenarios"]))
    report["requests_per_worker_set"] = int(report["total_http_requests"] / len(report["worker_counts"]))
    if filename == "hard_stress_before.json":
        report["test_name"] = "hard_stress_before"
    else:
        report["test_name"] = "hard_stress_after"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
