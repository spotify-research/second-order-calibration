#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python experiment_crowd_weather_disagreement.py
