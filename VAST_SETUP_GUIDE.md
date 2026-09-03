# Vast Runner Setup Tutorial

## A. Verify the local machine

Required local tools:

```bash
vastai --help
tmux -V
rsync --version
ssh -V
```

If all four work, the local command-line dependencies are present.

To verify Vast authentication separately:

```bash
vastai show instances
```

or:

```bash
vastai show user
```

---

## B. Install local dependencies on Ubuntu

If starting from scratch:

```bash
sudo apt update
sudo apt install -y git tmux rsync openssh-client pipx
pipx ensurepath
source ~/.bashrc
pipx install vastai
```

Configure Vast once:

```bash
vastai set api-key YOUR_VAST_API_KEY
vastai show instances
```

---

## C. Put the files in the repository

Expected project layout:

```text
SyncNetProject/
├── .env
├── .env.example
├── requirements.txt
├── .gitignore
├── bash_scripts/
│   ├── vast_worker
│   ├── vast_experiment_template
│   └── <your experiment campaigns>
└── scripts/
    └── vast/
        ├── vast_find.py
        ├── vast_run.py
        └── vast_sync_outputs.py
```

Make executable:

```bash
chmod +x bash_scripts/vast_worker
chmod +x bash_scripts/vast_experiment_template
chmod +x scripts/vast/vast_find.py
chmod +x scripts/vast/vast_run.py
chmod +x scripts/vast/vast_sync_outputs.py
```

---

## D. Create `.env`

From the project root:

```bash
cp .env.example .env
chmod 600 .env
```

Fill:

```dotenv
GITHUB_TOKEN=github_pat_...
WANDB_API_KEY=...

VAST_REPO_URL=https://github.com/NikWill2003/SyncNetProject.git
VAST_REPO_BRANCH=main
VAST_IMAGE=vastai/base-image:cuda-13.2-mini-py312-2026-08-26
```

Do not add the Vast API key here.

---

## E. `.gitignore`

Add:

```gitignore
.env
.vast/
bash_scripts/logs/
```

If `bash_scripts/logs/` was already tracked:

```bash
git rm -r --cached bash_scripts/logs
git commit -m "stop tracking runtime bash logs"
```

---

## F. Requirements

Merge `requirements.vast.snippet.txt` into the project's real `requirements.txt`.

The important PyTorch lines are:

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.12.0+cu130
```

The remote worker installs only:

```bash
python -m pip install -r requirements.txt
```

so `requirements.txt` is the Python-environment source of truth.

---

## G. Create an experiment campaign

Copy:

```bash
cp bash_scripts/vast_experiment_template bash_scripts/my_campaign
chmod +x bash_scripts/my_campaign
```

Edit the `RUNS=(...)` array.

Each entry is:

```text
"unique_name Hydra arguments..."
```

Example:

```bash
RUNS=(
  "soc_transformer_s0 task=sort_of_clevr experiment=sort_of_clevr/baselines/thesis/transformer train.seed=0"
  "soc_transformer_s1 task=sort_of_clevr experiment=sort_of_clevr/baselines/thesis/transformer train.seed=1"
)
```

---

## H. Push remote-required code

The Vast machine clones GitHub.

Before launching:

```bash
git add requirements.txt bash_scripts/vast_worker bash_scripts/my_campaign scripts/vast
git commit -m "prepare Vast campaign"
git push
```

The launcher also warns if tracked local changes or unpushed commits are detected.

---

## I. Search without renting

```bash
python scripts/vast/vast_find.py \
    --gpu "RTX 5090" \
    --tiers A,B,C \
    --top 20
```

---

## J. Launch

```bash
python scripts/vast/vast_run.py auto bash_scripts/my_campaign
```

Auto mode displays the top 20 candidates.

Choose a candidate number and confirm it.

After successful creation, the command returns to your normal shell while the detached tmux controller continues.

---

## K. Attach to the run

For:

```text
bash_scripts/my_campaign
```

attach:

```bash
tmux attach -t vast_my_campaign
```

Windows:

```text
run | ssh | btop | gpu
```

Useful tmux keys:

```text
Ctrl-b d   detach
Ctrl-b 0   run
Ctrl-b 1   ssh
Ctrl-b 2   btop
Ctrl-b 3   gpu
```

---

## L. What happens automatically

```text
instance boot
-> SSH ready
-> clone GitHub repo
-> install requirements.txt
-> verify CUDA
-> run campaign
-> sparse status in tmux
-> sync remote outputs/ to local outputs/
-> save tiny diagnostics in .vast/remote_logs/
-> append .vast/run_history.log
-> destroy Vast instance
```

If synchronization fails, the instance is retained.

---

## M. Recover after reboot

From the project root:

```bash
python scripts/vast/vast_run.py --resume-all
```

Resume one:

```bash
python scripts/vast/vast_run.py --resume my_campaign
```

Remote training continues even while your local PC is offline.

---

## N. Manual recovery sync

If needed:

```bash
python scripts/vast/vast_sync_outputs.py INSTANCE_ID --run-name my_campaign
```

---

## O. Recommended first test

Create one short campaign and launch with:

```bash
python scripts/vast/vast_run.py \
    auto \
    bash_scripts/vast_smoke_test \
    --gpu "RTX 5090" \
    --tiers A,B,C \
    --keep
```

Verify all four tmux windows, W&B authentication, sparse status, output synchronization, and run history. Then manually destroy the test instance and use normal auto-deletion for real campaigns.
