#!/usr/bin/env bash
set -euo pipefail

FILE=${1:-""}

if [ -z "$FILE" ]; then
    echo "Syncing all contents from notebooks/..."
    rclone copy notebooks/ pwr-remote:robustness/notebooks
else
    if [ -f "notebooks/$FILE" ]; then
        echo "Syncing specific file: $FILE"
        rclone copy "notebooks/$FILE" pwr-remote:robustness/notebooks
    else
        echo "Error: File 'notebooks/$FILE' does not exist."
        exit 1
    fi
fi

echo 'Colab notebooks sync successful'