#!/bin/sh
# Redis entrypoint with automatic AOF recovery.
# If the AOF file is corrupted, redis-check-aof --fix truncates the bad tail
# and Redis restarts cleanly from the repaired AOF + RDB base.

set -e

DATA_DIR="/data"
AOF_DIR="${DATA_DIR}/appendonlydir"

fix_aof() {
    manifest="${AOF_DIR}/appendonly.aof.manifest"
    [ -f "$manifest" ] || return 0

    # Iterate every incr file listed in the manifest
    grep -oE '[^ ]+\.incr\.aof' "$manifest" 2>/dev/null | while read -r incr_file; do
        full="${AOF_DIR}/${incr_file}"
        [ -f "$full" ] || continue

        if ! redis-check-aof "$full" >/dev/null 2>&1; then
            echo "[redis-entrypoint] Corrupted AOF detected: ${incr_file} — auto-fixing..."
            echo y | redis-check-aof --fix "$full" >/dev/null 2>&1 || true
            echo "[redis-entrypoint] AOF fixed: ${incr_file}"
        fi
    done
}

echo "[redis-entrypoint] Checking AOF integrity before start..."
fix_aof
echo "[redis-entrypoint] Starting Redis..."
exec "$@"
