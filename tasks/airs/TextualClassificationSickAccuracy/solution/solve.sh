#!/usr/bin/env bash
# Reference solution for TextualClassificationSickAccuracy, run by Harbor's `oracle` agent from /solution.
# Deliberately far from SOTA: it exists to prove the grader responds.
set -euo pipefail
cd /app
python3 /solution/reference.py
test -f /app/submission.csv
wc -l /app/submission.csv
