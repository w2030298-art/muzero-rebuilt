#!/usr/bin/env bash
set -euo pipefail
uv run muzero benchmark --config configs/cartpole_muzero.yaml --profile laptop_rtx4060_8gb --component inference --steps 100
uv run muzero benchmark --config configs/cartpole_muzero.yaml --profile laptop_rtx4060_8gb --component search --steps 20
