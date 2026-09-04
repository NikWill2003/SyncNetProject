#!/usr/bin/env bash
# Shared campaign runner: turns a RUNS array into a GPU-parallel campaign.
#
# A campaign script defines:
#   CAMPAIGN   (basename), LOG (log dir)
#   PREP       optional: hydra args for scripts.prepare_dataset. Runs ONCE,
#              serially, BEFORE any GPU worker starts -- N workers must never
#              race to generate the same dataset.
#   RUNS       array of "name arg arg arg ..."
# then calls: run_campaign
#
# Scheduling is a SHARED QUEUE, not static sharding: whichever GPU finishes
# first takes the next run, so uneven runtimes load-balance themselves.
# Contract preserved for vast_worker: one "ok <name>" / "FAIL <name>" line
# per run, per-run log at $LOG/<name>.log.
#
#   NGPU=n   override the detected GPU count (default: nvidia-smi -L | wc -l)
#   SKIP_PREP=1  skip the prepare step

run_campaign() {
    local n_gpu="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
    [ "${n_gpu:-0}" -ge 1 ] 2>/dev/null || n_gpu=1

    echo "[campaign] $CAMPAIGN: ${#RUNS[@]} runs on ${n_gpu} GPU(s)"

    # --- serial prepare phase (before any worker) --------------------
    if [ -n "${PREP:-}" ] && [ "${SKIP_PREP:-0}" != 1 ]; then
        echo "[campaign] preparing dataset: $PREP"
        if python -u -m scripts.prepare_dataset $PREP > "$LOG/_prepare.log" 2>&1; then
            echo "ok   _prepare"
        else
            echo "FAIL _prepare exit=$? ($LOG/_prepare.log)"
            return 1
        fi
    fi

    # --- workers per GPU: small models leave a big GPU idle, so more than
    #     one worker can share a device. Each worker is its own process with
    #     its own copy of the gpu_cached dataset (~13 GB on sqoop), so 2/GPU
    #     needs a 32 GB card. Set in the campaign or via the env: WORKERS_PER_GPU.
    local wpg="${WORKERS_PER_GPU:-1}"
    [ "${wpg:-1}" -ge 1 ] 2>/dev/null || wpg=1
    local n_workers=$(( n_gpu * wpg ))
    [ "$wpg" -gt 1 ] && echo "[campaign] $wpg workers per GPU -> $n_workers workers"

    # --- fair CPU split: N workers must not each claim every core ----
    local cores; cores=$(nproc 2>/dev/null || echo 8)
    local threads=$(( cores / n_workers )); [ "$threads" -lt 1 ] && threads=1
    export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"

    if [ "$n_workers" -le 1 ]; then
        _campaign_worker 0 0 1
        return $?
    fi

    # --- shared queue: a claim file per run, taken atomically --------
    local qdir="$LOG/.queue"; rm -rf "$qdir"; mkdir -p "$qdir"
    local w pids=()
    for (( w = 0; w < n_workers; w++ )); do
        _campaign_worker "$(( w % n_gpu ))" "$w" "$n_workers" &
        pids+=($!)
        sleep "${STAGGER:-10}"
    done
    local rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}

_campaign_worker() {  # <gpu> <worker-id> <n_workers>
    local gpu=$1 wid=$2 nw=$3 i spec name failed=0
    local qdir="$LOG/.queue"
    for i in "${!RUNS[@]}"; do
        if [ "$nw" -gt 1 ]; then
            # atomic claim: mkdir succeeds for exactly one worker
            mkdir "$qdir/$i" 2>/dev/null || continue
        fi
        spec="${RUNS[$i]}"
        read -r -a parts <<< "$spec"
        name="${parts[0]}"
        args=("${parts[@]:1}")

        local ok=0
        if [ "$nw" -gt 1 ]; then
            # N workers share one stdout: writing training output there would
            # interleave N streams and can corrupt the ok/FAIL line contract.
            echo "RUNNING $name (gpu $gpu)"
            CUDA_VISIBLE_DEVICES="$gpu" \
                python -u -m scripts.main "${args[@]}" \
                > "$LOG/$name.log" 2>&1 || ok=$?
        else
            # NOTE: no "|| true" here -- it would run as the last command and
            # reset PIPESTATUS to (0), silently turning every failure into ok.
            # set -o pipefail (no -e) already keeps a failed run non-fatal.
            CUDA_VISIBLE_DEVICES="$gpu" \
                python -u -m scripts.main "${args[@]}" \
                2>&1 | tee "$LOG/$name.log"
            ok=${PIPESTATUS[0]}
        fi
        if [ "$ok" -eq 0 ]; then
            echo "ok   $name"
        else
            echo "FAIL $name exit=$ok ($LOG/$name.log)"
            failed=1
        fi
    done
    return "$failed"
}
