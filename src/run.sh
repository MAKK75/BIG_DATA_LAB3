#!/bin/bash
export GIT_PYTHON_REFRESH=quiet

python3 /app/app.py > /app/logs/output.log 2>&1