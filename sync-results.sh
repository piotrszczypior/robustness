#!/usr/bin/env bash
set -euo pipefail

# 'pull' (download from GD)
# 'push' (upload to GD)
ACTION=${1:-"pull"}

FILE=${2:-""}

REMOTE="pwr-remote:robustness/results"
LOCAL="results/"

if [ "$ACTION" == "pull" ]; then
    if [ -z "$FILE" ]; then
        echo "Pulling files from Google Drive to local $LOCAL/..."
        rclone copy "$REMOTE" "$LOCAL" --ignore-existing -P
    else
        echo "Pulling specific file: $FILE from Google Drive..."
        rclone copy "$REMOTE/$FILE" "$LOCAL" --ignore-existing -P
    fi

elif [ "$ACTION" == "push" ]; then
    if [ -z "$FILE" ]; then
        echo "Pushing local files from $LOCAL/ to Google Drive..."
        rclone copy "$LOCAL" "$REMOTE" --ignore-existing -P
    else
        if [ -f "$LOCAL/$FILE" ]; then
            echo "Pushing specific file: $FILE to Google Drive..."
            rclone copy "$LOCAL/$FILE" "$REMOTE" --ignore-existing -P
        else
            echo "Error: File '$LOCAL/$FILE' does not exist locally."
            exit 1
        fi
    fi

else
    echo "Error: Unknown action '$ACTION'. Please use 'pull' or 'push'."
    echo "Usage: ./sync_results.sh [pull|push] [filename.csv]"
    exit 1
fi

echo "rclone operation completed successfully!"