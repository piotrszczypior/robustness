#!/usr/bin/env bash
set -euo pipefail

# 'pull' (download from GD)
# 'push' (upload to GD)
ACTION=${1:-"pull"}

TARGET=${2:-"results"}

PATTERN=${3:-""}


if [ "$TARGET" == "results" ]; then
    REMOTE="pwr-remote:robustness/results"
    LOCAL="results/"
elif [ "$TARGET" == "images" ]; then
    REMOTE="pwr-remote:robustness/images"
    LOCAL="images/"
else
    echo "Error: Unknown target '$TARGET'. Please use 'results' or 'images'."
    echo "Usage: $0 [pull|push] [results|images] [pattern]"
    exit 1
fi

RCLONE_CMD="rclone copy --ignore-existing -P"

if [ "$ACTION" == "pull" ]; then
    if [ -z "$PATTERN" ]; then
        echo "Pulling all files from Google Drive to local $LOCAL/..."
        $RCLONE_CMD "$REMOTE" "$LOCAL"
    else
        echo "Pulling files matching '$PATTERN' from Google Drive..."
        $RCLONE_CMD "$REMOTE" "$LOCAL" --include "$PATTERN"
    fi

elif [ "$ACTION" == "push" ]; then
    if [ -z "$PATTERN" ]; then
        echo "Pushing all local files from $LOCAL/ to Google Drive..."
        $RCLONE_CMD "$LOCAL" "$REMOTE"
    else
        echo "Pushing local files matching '$PATTERN' to Google Drive..."
        $RCLONE_CMD "$LOCAL" "$REMOTE" --include "$PATTERN"
    fi

else
    echo "Error: Unknown action '$ACTION'. Please use 'pull' or 'push'."
    echo "Usage: ./sync_results.sh [pull|push] [filename.csv]"
    exit 1
fi

echo "rclone operation completed successfully!"