# Vast Runner Schema

## 1. Goal

Run independent, disposable, single-GPU SyncNet experiment campaigns on Vast.ai with:

- automatic marketplace filtering and ranking;
- one experiment campaign per local tmux session;
- persistent local tracking across PC reboots;
- sparse remote progress reporting;
- direct SSH, `btop`, and `nvidia-smi` tmux windows;
- automatic `outputs/` synchronization;
- one-line local run history;
- automatic instance deletion only after synchronization succeeds.

Normal launch:

```bash
python scripts/vast/vast_run.py auto bash_scripts/explore1
```

Recovery after restarting the local PC:

```bash
python scripts/vast/vast_run.py --resume-all
```

---

## 2. CPU-selection policy

### Why CPU performance is deliberately important

The SyncNet workloads are relatively small/fine-grained GPU training jobs. Empirically, this workload behaved very differently from a large compute-heavy Transformer:

- a 5090 was faster than a 4090 on a large synthetic Transformer;
- the same 5090 host was much slower than the local 4090 on the small SyncNet/Sort-of-CLEVR training loop;
- the poor host used an older server CPU, while the local system used a Ryzen 9 9950X.

That points to CPU/driver/kernel-launch latency and other fine-grained host-side overheads being important for this workload. Therefore the marketplace policy prioritizes strong single-thread CPU performance instead of simply maximizing server core count.

### Reference and tiers

Reference:

```text
Ryzen 9 9950X = 4,727 PassMark single-thread = 100%
```

Tiers:

```text
A  >= 95%    >= 4,491
B  90–95%     4,255–4,490
C  85–90%     4,018–4,254
D  80–85%     3,782–4,017
E  70–80%     3,309–3,781
```

Normal automatic runs use:

```text
A,B,C
```

That means the host CPU must have a known single-thread score of at least about 85% of the 9950X reference unless the user explicitly widens `--tiers`.

### Other default marketplace filters

The finder also prefers/requires:

```text
GPU count       = 1
GPU             = RTX 4090 or RTX 5090
Reliability     >= 0.99
Effective CPUs  >= 8
Direct SSH      required
Disk            >= requested disk (default 200 GB)
Duration        >= 1 day by default
```

### Ranking order

Among eligible offers:

1. higher known single-thread CPU score;
2. RTX 5090 ahead of RTX 4090;
3. lower hourly price;
4. higher reliability.

The aim is not to claim PassMark perfectly predicts training throughput. It is a low-friction host-selection proxy that strongly avoids the class of slow old-server hosts already observed in this workload.

---

## 3. File/component schema

```text
SyncNetProject/
├── .env
├── .env.example
├── requirements.txt
├── outputs/
├── .vast/
│   ├── active/
│   ├── remote_logs/
│   └── run_history.log
├── bash_scripts/
│   ├── vast_worker
│   ├── vast_experiment_template
│   ├── explore1
│   └── logs/
└── scripts/
    └── vast/
        ├── vast_find.py
        ├── vast_run.py
        └── vast_sync_outputs.py
```

### `scripts/vast/vast_find.py`

Purpose:

```text
Vast marketplace -> filter -> CPU score/tier -> rank -> display
```

It never rents or destroys an instance.

### `scripts/vast/vast_run.py`

Main local orchestrator.

Responsibilities:

```text
load .env
git/push preflight
show top 20 auto candidates
select/create instance
persist .vast/active state
create detached tmux controller
monitor startup/run
sync outputs + diagnostics
append one-line history
destroy after successful sync
resume one/all tracked runs
```

### `scripts/vast/vast_sync_outputs.py`

Reusable synchronization helper.

Copies:

```text
remote /workspace/SyncNetProject/outputs/
    ->
local ./outputs/
```

and tiny infrastructure diagnostics into:

```text
.vast/remote_logs/<campaign>/<instance-id>/
```

### `bash_scripts/vast_worker`

Remote worker cloned from GitHub.

Responsibilities:

```text
pip install requirements.txt
verify torch/CUDA/GPU/W&B environment
launch requested Bash campaign
emit sparse status
emit five-minute heartbeat
write RUN_SUCCESS or RUN_FAILED
```

### Experiment campaign

Example:

```text
bash_scripts/explore1
```

Contract:

```text
log path: bash_scripts/logs/<campaign>/<run-name>.log
success line: ok   <run-name>
failure line: FAIL <run-name> ...
campaign exit: 0 only if all intended runs succeeded
```

---

## 4. Tmux schema

One Bash campaign maps to exactly one local tmux session:

```text
bash_scripts/explore1 -> vast_explore1
```

The same campaign cannot be launched twice concurrently.

Each active session contains four windows:

```text
run | ssh | btop | gpu
```

`run`
: Sparse startup, remote status, synchronization, and cleanup events.

`ssh`
: Direct interactive SSH shell to the exact Vast instance.

`btop`
: Remote CPU/RAM/process monitoring.

`gpu`
: Remote `watch -n 1 nvidia-smi`.

The foreground `vast_run.py` command returns after the instance is created and the detached tmux controller is established.

---

## 5. Lifecycle/state machine

```text
LOCAL
  |
  | vast_run auto bash_scripts/explore1
  v
git preflight
  |
  v
show top 20 candidates
  |
  v
user selects offer
  |
  v
create Vast instance
  |
  v
write .vast/active/explore1.json
  |
  v
start vast_explore1 tmux
  |
  +------------------------> foreground shell returns
  |
REMOTE/CONTROLLER
  |
  v
wait for SSH
  |
  v
clone repo
  |
  v
install requirements
  |
  v
verify GPU
  |
  v
run campaign
  |
  +--> per-run sparse RUNNING / COMPLETED / FAILED
  |
  v
campaign ends
  |
  v
sync outputs + diagnostics
  |
  +--> sync fails -> KEEP instance + active state
  |
  v
append one-line history
  |
  v
destroy instance
  |
  v
remove active state + tmux session
```

---

## 6. Recovery schema

The remote training process is independent of the local controller.

If the PC/tmux/internet disappears:

```text
remote training continues
```

Persistent local state remains in:

```text
.vast/active/*.json
```

After reboot:

```bash
python scripts/vast/vast_run.py --resume-all
```

The launcher recreates missing tracking sessions. If a remote campaign already finished, the restored controller immediately performs sync/history/deletion.

---

## 7. Output/data policy

Canonical experiment data:

```text
outputs/
```

This is synchronized at campaign end.

Small controller diagnostics:

```text
.vast/remote_logs/
```

One-line audit/history:

```text
.vast/run_history.log
```

Deletion rule:

```text
sync success -> delete
sync failure -> retain instance
```

This is the key data-safety invariant.

---

## 8. Secrets policy

Project root:

```text
.env
```

contains:

```text
GITHUB_TOKEN
WANDB_API_KEY
```

plus optional non-secret Vast defaults.

`.env` is local-only and ignored by Git.

The generic Vast API key is not transferred to the worker. The local `vastai` CLI uses its own stored API key for marketplace/search/create/destroy operations.

---

## 9. Experiment-campaign schema

Minimal pattern:

```bash
#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

CAMPAIGN="$(basename "$0")"
LOG="bash_scripts/logs/$CAMPAIGN"
mkdir -p "$LOG"

RUNS=(
  "run_name task=... experiment=... train.seed=0"
)

FAILED=0

for spec in "${RUNS[@]}"; do
    read -r -a parts <<< "$spec"
    name="${parts[0]}"
    args=("${parts[@]:1}")

    if CUDA_VISIBLE_DEVICES=0 \
        python -u -m scripts.main "${args[@]}" \
        2>&1 | tee "$LOG/$name.log"
    then
        echo "ok   $name"
    else
        code=${PIPESTATUS[0]}
        echo "FAIL $name exit=$code ($LOG/$name.log)"
        FAILED=1
    fi
done

exit "$FAILED"
```

Why this contract:

- each sub-run gets a stable human-readable name;
- file creation gives the controller a sparse `RUNNING` event;
- `ok` / `FAIL` gives sparse completion state;
- a single sub-run failure does not necessarily abort all later experiments;
- the final exit status still accurately marks the entire campaign.

---

## 10. Multi-GPU boxes (added)

A multi-GPU host is treated as **N independent single-GPU workers**, never as
one distributed job. Every experiment still uses exactly one GPU.

```text
one instance
  |
  +-- serial PREP phase (dataset built ONCE)
  |
  +-- GPU 0 --\
  +-- GPU 1 ---+--> shared queue of runs (work-stealing)
  +-- GPU N ---/
```

### `bash_scripts/_campaign_lib.sh`

Every campaign now ends with:

```bash
source bash_scripts/_campaign_lib.sh
run_campaign
```

and declares `PREP` (hydra args for one `scripts.prepare_dataset` call) plus
`RUNS`. `run_campaign`:

1. detects GPUs (`NGPU` env overrides; default `nvidia-smi -L | wc -l`);
2. runs `PREP` **serially, before any worker starts** — N workers must never
   race to generate the same dataset (`SKIP_PREP=1` skips it);
3. caps `OMP_NUM_THREADS`/`MKL_NUM_THREADS` at `nproc / NGPU`;
4. starts one worker per GPU, staggered by `STAGGER` (default 10 s);
5. dispatches from a **shared queue**, not static shards: claims are
   `mkdir "$LOG/.queue/<i>"`, which is atomic, so each run executes exactly
   once and a fast GPU immediately takes the next job;
6. preserves the status contract — one `ok <name>` / `FAIL <name>` line per
   run, per-run log at `$LOG/<name>.log`.

The same campaign file therefore runs unchanged on 1, 2, 4, or 8 GPUs.

### Launcher

```bash
python scripts/vast/vast_run.py auto bash_scripts/run_soc_1_assembly --num-gpus 4
```

`--num-gpus N` filters the marketplace to exactly N-GPU boxes and passes
`NGPU=N` to the instance. `--min-cpus-per-gpu` (default 8) enforces CPU
*quantity* per GPU independently of the single-thread *quality* tiers, so a
64-core 8-GPU box with weak cores is still rejected. Offers are ranked by
single-thread score, GPU model, **$/GPU/h**, effective **cores/GPU**, then
reliability; shared quantities (price, cores, disk, bandwidth) are normalised
per GPU while single-thread score, GHz, VRAM, reliability and PCIe are not.

### Periodic sync

`--sync-every SECONDS` (default 3600, `0` disables) makes the controller pull
`outputs/` and the campaign logs mid-run. It is non-fatal and never destroys
the instance; the destroy-only-after-successful-sync invariant is unchanged.
The cadence is persisted in the active-state JSON, so `--resume` keeps it.


---

## 11. Boot failures: blacklist + auto-retry (added)

Hosts fail. Two real cases seen: a machine whose container runtime cannot
start the image at all (`failed to start containers`), and a machine that
boots but never reaches the campaign.

The controller now:

1. **detects** fatal host errors during the SSH wait by scanning
   `vastai logs` for known-fatal patterns, instead of waiting out the clock;
2. **aborts** at `--startup-deadline` (default 1800 s) if the campaign never
   reaches RUNNING, saving `instance_boot.log` and destroying the box;
3. **blacklists the MACHINE** (not the offer) in `.vast/blacklist.json` --
   a broken host fails identically on every offer it lists;
4. **re-rents automatically**: `--auto-retry N` (default 2) picks the
   *cheapest tier-A box under `--max-price-per-gpu`* (default $0.70/GPU/h),
   falling back to B then C, always skipping blacklisted machines. If no
   eligible offer exists, the campaign fails cleanly.

`--min-cuda` filters hosts whose driver is older than the image needs; a
CUDA 13.x image on an older host is a plausible cause of container-start
failures, so match it to your image (e.g. `--min-cuda 13.0`).
