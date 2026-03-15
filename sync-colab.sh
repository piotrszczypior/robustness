#!/usr/bin/env bash
set -euo pipefail

rclone copy notebooks/ pwr-remote:robustness/notebooks
echo 'Colab notebooks sync successful'