#!/usr/bin/env bash
set -euo pipefail

START_TS=$(date +%s)
MAX_SECONDS=$((16 * 60 * 60))   # 16 hours
SLEEP_SECONDS=60               # 1 minute between cycles

ts() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] [INFO] Starting incremental copy+scrub loop"
echo "[$(ts)] [INFO] Will stop after $((MAX_SECONDS/3600)) hours"

while true; do
    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))

    if [ "$ELAPSED" -ge "$MAX_SECONDS" ]; then
        echo "[$(ts)] [INFO] Reached $((MAX_SECONDS/3600))-hour limit; exiting."
        break
    fi

    echo
    echo "[$(ts)] [INFO] ===== elapsed=$((ELAPSED/60)) min ====="

    echo "[$(ts)] [INFO] Running copy_implicated_shards.py"
    python scripts/copy_implicated_shards.py \
      --map missing-taxa.csv \
      --src-dir /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards \
      --dst-dir /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards_to_scrub \
      --scrubbed-dir /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards_scrubbed

    echo "[$(ts)] [INFO] Running remove_bad_samples_parallel.py"
    python scripts/remove_bad_samples_parallel.py \
      --shards /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards_to_scrub \
      --shards-scrubbed /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards_scrubbed \
      --map missing-taxa.csv \
      --workers 2 \
      --max-inflight 2

    echo "[$(ts)] [INFO] Publishing scrubbed shards back into primary shards/"
    python scripts/publish_scrubbed_shards.py \
      --shards /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards \
      --shards-scrubbed /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards_scrubbed

    echo "[$(ts)] [INFO] Updating sizes.json"
    python scripts/sizes_incremental_fast.py \
      --shards /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/2025-12_common-name-fix/shards \
      --map missing-taxa.csv \
      --min-age-seconds 180 \
      --progress-every 50 \
      --heartbeat-seconds 30

    echo "[$(ts)] [INFO] Sleeping ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
done

echo "[$(ts)] [DONE] Incremental loop finished"
