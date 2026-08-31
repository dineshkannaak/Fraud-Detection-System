#!/bin/sh
set -eu

python scripts/fetch_artifacts.py
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
