#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vast_find import lookup_offer, print_candidate, print_table, search_offers
from vast_sync_outputs import sync_instance

DEFAULT_IMAGE = "vastai/pytorch:cuda-12.8.1-auto"

# Bumped whenever runner behaviour changes, so `--version` can prove which
# code is actually running (the box clones the repo; your local copy may lag).
RUNNER_VERSION = "2026-09-03.2"
RUNNER_FEATURES = (
    "direct-ssh-endpoint", "verified-destroy", "startup-watchdog",
    "fatal-boot-detection", "machine-blacklist", "auto-retry-tier-ladder",
    "periodic-sync", "quiet-ssh", "multi-gpu-any-count", "repo-content-preflight",
    "destroy-prompt-answered",
)
DEFAULT_REPO = "https://github.com/NikWill2003/SyncNetProject.git"
DEFAULT_BRANCH = "main"
REMOTE_WORKDIR = "/workspace/SyncNetProject"


def log(stage: str, message: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {stage:<12} {message}", flush=True)


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(cmd, text=True, capture_output=capture)
    if check and p.returncode:
        raise RuntimeError((p.stderr or p.stdout or "").strip())
    return p


def parse_machine_output(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None


def parse_instance_id(raw: str) -> int:
    data = parse_machine_output(raw)
    if isinstance(data, dict) and data.get("new_contract"):
        return int(data["new_contract"])
    raise RuntimeError(f"Could not parse new instance ID from:\n{raw}")


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Copy .env.example to .env and fill it in.")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise SystemExit(f"Invalid .env line: {raw}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            print(f"WARNING: {key} is already set in your shell and differs from "
                  f".env; the SHELL value wins (os.environ.setdefault). "
                  f"`unset {key}` if .env is the one you want.")
        os.environ.setdefault(key, value)


def preflight_repo(repo: str, branch: str, token: str) -> None:
    """Prove the token can read this repo/branch BEFORE renting anything.
    A bad or expired token otherwise surfaces as `git clone` HTTP 401 on the
    instance, minutes and cents later."""
    if not repo.startswith("https://"):
        return
    authed = repo.replace("https://", f"https://x-access-token:{token}@", 1)
    p = subprocess.run(["git", "ls-remote", "--heads", authed, branch],
                       text=True, capture_output=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip().replace(token, "<token>")
        raise SystemExit(
            f"Repo preflight FAILED for {repo} (branch {branch}):\n  {err}\n"
            "  401 => token expired/revoked, or a fine-grained PAT without "
            "'Contents: Read' on this repo, or a stale GITHUB_TOKEN exported "
            "in your shell overriding .env.")
    if not p.stdout.strip():
        raise SystemExit(
            f"Repo preflight FAILED: branch {branch!r} does not exist on {repo}. "
            "Push it, or set VAST_REPO_BRANCH in .env.")
    print(f"preflight ok: {repo} branch {branch} readable")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is missing from .env")
    if any(c.isspace() for c in value):
        raise SystemExit(f"{name} contains whitespace, which this launcher does not support")
    return value


def run_name(run_script: str) -> str:
    stem = Path(run_script).stem
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "run"


def session_name(run_script: str) -> str:
    return f"vast_{run_name(run_script)}"


def active_dir(root: Path) -> Path:
    return root / ".vast" / "active"


def active_path(root: Path, run_script: str) -> Path:
    return active_dir(root) / f"{run_name(run_script)}.json"


def atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Could not read active state {path}: {e}") from e


def update_state(path: Path, **changes: Any) -> dict[str, Any]:
    state = load_state(path)
    state.update(changes)
    atomic_json_write(path, state)
    return state


def tmux_exists(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def tmux_windows(session: str) -> set[str]:
    if not tmux_exists(session):
        return set()
    p = run(["tmux", "list-windows", "-t", session, "-F", "#{window_name}"], check=False)
    return set(p.stdout.splitlines()) if p.returncode == 0 else set()


def history_path(root: Path) -> Path:
    return root / ".vast" / "run_history.log"


def already_succeeded(root: Path, script: str) -> bool:
    p = history_path(root)
    if not p.is_file():
        return False
    needle = f"| success | {script} |"
    return any(needle in line for line in p.read_text().splitlines())


def duration_text(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def append_history(root: Path, status: str, script: str, offer: dict[str, Any], instance_id: int, runtime: float) -> None:
    p = history_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    rate = float(offer.get("price") or 0.0)
    cost = rate * runtime / 3600 if rate > 0 else 0.0
    cost_s = f"~${cost:.2f}" if rate > 0 else "$?"
    rate_s = f"${rate:.3f}/h" if rate > 0 else "$?/h"
    pct = offer.get("st_pct")
    pct_s = f"{pct:.1f}%" if pct is not None else "?"
    line = (
        f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')} | {status} | {script} | "
        f"{offer.get('gpu', '?')} | {offer.get('cpu', '?')} | ST={pct_s} | "
        f"{duration_text(runtime)} | {cost_s} | {rate_s} | "
        f"offer={offer.get('id', '?')} | instance={instance_id}\n"
    )
    with p.open("a") as f:
        f.write(line)


def git_preflight(root: Path, branch: str, run_script_rel: str | None = None) -> None:
    if not (root / ".git").exists():
        return
    problems: list[str] = []
    dirty = run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], check=False)
    if dirty.returncode == 0 and dirty.stdout.strip():
        problems.append("tracked files have uncommitted changes")
    fetch = run(["git", "-C", str(root), "fetch", "--quiet", "origin", branch], check=False)
    if fetch.returncode == 0:
        ahead = run(["git", "-C", str(root), "rev-list", "--count", "FETCH_HEAD..HEAD"], check=False)
        try:
            if ahead.returncode == 0 and int(ahead.stdout.strip() or "0") > 0:
                problems.append(f"local HEAD has {ahead.stdout.strip()} commit(s) not on origin/{branch}")
        except ValueError:
            pass
    else:
        problems.append(f"could not verify origin/{branch}")
    # Content check: the box runs whatever is in origin/<branch>. Comparing
    # commits is not enough (wrong branch, unpushed work, a stale remote all
    # look fine). Compare the ACTUAL bytes of the files that drive the run.
    critical = ["bash_scripts/_campaign_lib.sh", "bash_scripts/vast_worker"]
    if run_script_rel:
        critical.insert(0, run_script_rel)
    for rel in critical:
        local = root / rel
        if not local.is_file():
            continue
        shown = run(["git", "-C", str(root), "show", f"origin/{branch}:{rel}"],
                    check=False)
        if shown.returncode != 0:
            problems.append(f"{rel} does not exist on origin/{branch}")
        elif shown.stdout != local.read_text():
            problems.append(f"{rel} DIFFERS from origin/{branch} "
                            f"(the box would run the remote version)")

    if not problems:
        return
    print("\nWARNING: the Vast box clones the remote Git repo, not your local working tree.")
    for problem in problems:
        print(f"  - {problem}")
    if input("Continue anyway? [y/N] ").strip().lower() not in {"y", "yes"}:
        raise SystemExit("Cancelled.")


def choose_auto(args: argparse.Namespace) -> dict[str, Any]:
    tiers = {x.strip().upper() for x in args.tiers.split(",") if x.strip()}
    offers = search_offers(
        gpus=args.gpu or None,
        max_price=args.max_price,
        min_reliability=args.min_reliability,
        min_cpus=args.min_cpus,
        min_disk=args.disk,
        exact_gpus=(None if args.num_gpus in (0, None) else args.num_gpus),
        verified_only=args.verified_only,
        min_cpus_per_gpu=args.min_cpus_per_gpu,
        min_duration=args.min_duration,
        allowed_tiers=tiers,
        limit=args.search_limit,
    )
    if not offers:
        raise RuntimeError(f"No candidates in tiers {args.tiers}.")
    candidates = offers[:args.top]
    print(f"\nTop {len(candidates)} Vast candidates:\n")
    print_table(candidates)
    if args.yes:
        return candidates[0]
    while True:
        answer = input(f"Select candidate [1-{len(candidates)}], or [q]uit: ").strip().lower()
        if answer in {"q", "quit"}:
            raise SystemExit("Cancelled.")
        try:
            index = int(answer)
        except ValueError:
            print("Enter a candidate number or q.")
            continue
        if not 1 <= index <= len(candidates):
            print(f"Enter a number from 1 to {len(candidates)}.")
            continue
        offer = candidates[index - 1]
        print()
        print_candidate(offer)
        if input("Use this box? [Y/n]: ").strip().lower() in {"", "y", "yes"}:
            return offer
        print()
        print_table(candidates)


def get_ssh(instance_id: int) -> tuple[str, str, int]:
    p = run(["vastai", "ssh-url", str(instance_id)])
    m = re.search(r"ssh://([^@\s]+)@([^:\s/]+):(\d+)", p.stdout)
    if not m:
        raise RuntimeError(f"Could not parse ssh-url output: {p.stdout!r}")
    return m.group(1), m.group(2), int(m.group(3))


def ssh_command(user: str, host: str, port: int, remote: str | None = None) -> list[str]:
    cmd = [
        "ssh", "-tt", "-q", "-o", "LogLevel=ERROR", "-o", "ClearAllForwardings=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
        "-p", str(port), f"{user}@{host}",
    ]
    if remote is not None:
        cmd.append(remote)
    return cmd


def get_ssh_direct(instance_id: int) -> tuple[str, str, int] | None:
    """Direct SSH endpoint (public IP + the host port mapped to 22).

    `vastai ssh-url` returns the PROXY endpoint (sshN.vast.ai), which only
    works if the instance's reverse tunnel binds -- the failure mode that
    shows up as "remote port forwarding failed for listen port NNNN" and
    leaves us waiting for SSH forever. We rent with --direct, so use it."""
    p = run(["vastai", "show", "instance", str(instance_id), "--raw"], check=False)
    if p.returncode != 0:
        return None
    try:
        data = parse_machine_output(p.stdout)
    except Exception:
        return None
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), None)
    if not isinstance(data, dict):
        return None
    host = data.get("public_ipaddr") or data.get("ssh_host")
    ports = data.get("ports") or {}
    port = None
    if isinstance(ports, dict):
        mapping = ports.get("22/tcp") or ports.get("22/TCP")
        if isinstance(mapping, list) and mapping:
            port = mapping[0].get("HostPort") or mapping[0].get("hostport")
    if port is None:
        port = data.get("direct_port_start")
    if not host or port in (None, ""):
        return None
    try:
        port = int(port)
    except (TypeError, ValueError):
        return None
    if port <= 0:            # -1 means "not mapped yet" while the box boots
        return None
    return "root", str(host).strip(), port


# Host-side failures that no amount of waiting will fix. Seen in the wild:
# containerd cannot start the container at all, so SSH never appears.
FATAL_BOOT_PATTERNS = (
    "failed to start containers",
    "failed to create shim task",
    "failed to retrieve OCI runtime container pid",
    "no space left on device",
    "nvidia-container-cli: initialization error",
)


def instance_logs(instance_id: int, tail: int = 200) -> str:
    p = run(["vastai", "logs", str(instance_id), "--tail", str(tail)], check=False)
    return (p.stdout or "") + (p.stderr or "") if p.returncode == 0 else ""


def fatal_boot_error(instance_id: int) -> str | None:
    text = instance_logs(instance_id)
    for pat in FATAL_BOOT_PATTERNS:
        if pat in text:
            for line in reversed(text.splitlines()):
                if pat in line:
                    return line.strip()[:180]
            return pat
    return None


def instance_status(instance_id: int) -> str:
    p = run(["vastai", "show", "instance", str(instance_id), "--raw"], check=False)
    if p.returncode != 0:
        return "?"
    try:
        data = parse_machine_output(p.stdout)
    except Exception:
        return "?"
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), {})
    if not isinstance(data, dict):
        return "?"
    return str(data.get("actual_status") or data.get("cur_state") or "?")


def wait_for_ssh(instance_id: int, timeout: int) -> tuple[str, str, int]:
    deadline = time.time() + timeout
    next_notice = 0.0
    last = ""
    while time.time() < deadline:
        try:
            endpoint = get_ssh_direct(instance_id)
            kind = "direct"
            if endpoint is None:
                endpoint = get_ssh(instance_id)
                kind = "proxy"
            user, host, port = endpoint
            p = run([
                "ssh", "-q", "-o", "LogLevel=ERROR", "-o", "ClearAllForwardings=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8",
                "-p", str(port), f"{user}@{host}", "true",
            ], check=False)
            if p.returncode == 0:
                log("SSH_ENDPOINT", f"{kind}: {user}@{host}:{port}")
                return user, host, port
            last = (p.stderr or p.stdout).strip()
        except Exception as e:
            last = str(e)
        if time.time() >= next_notice:
            fatal = fatal_boot_error(instance_id)
            if fatal:
                raise RuntimeError(f"host cannot start the container: {fatal}")
            log("BOOTING", f"waiting for SSH (instance status: {instance_status(instance_id)}"
                           f"{'; last: ' + last[:90] if last else ''})")
            next_notice = time.time() + 30
        time.sleep(4)
    raise RuntimeError(f"Timed out waiting for SSH. Last error: {last}")


def ensure_monitor_windows(session: str, user: str, host: str, port: int) -> None:
    windows = tmux_windows(session)
    specs = [
        ("ssh", ssh_command(user, host, port)),
        ("btop", ssh_command(user, host, port, "while ! command -v btop >/dev/null 2>&1; do echo 'waiting for btop...'; sleep 3; done; exec btop")),
        ("gpu", ssh_command(user, host, port, "while ! command -v nvidia-smi >/dev/null 2>&1; do echo 'waiting for nvidia-smi...'; sleep 3; done; exec watch -n 1 nvidia-smi")),
    ]
    for name, cmd in specs:
        if name not in windows:
            subprocess.run(["tmux", "new-window", "-d", "-t", session, "-n", name, shlex.join(cmd)], check=True)


def remote_state(user: str, host: str, port: int) -> str:
    cmd = (
        "if [ -e /workspace/RUN_SUCCESS ]; then echo SUCCESS; "
        "elif [ -e /workspace/RUN_FAILED ]; then echo FAILED; "
        "elif [ -e /workspace/READY ]; then echo RUNNING; "
        "else echo BOOTSTRAP; fi"
    )
    p = run(["ssh", "-q", "-o", "LogLevel=ERROR", "-o", "ClearAllForwardings=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), f"{user}@{host}", cmd], check=False)
    return p.stdout.strip() if p.returncode == 0 else "DISCONNECTED"


def status_lines(user: str, host: str, port: int, start: int) -> list[str]:
    p = run([
        "ssh", "-q", "-o", "LogLevel=ERROR", "-o", "ClearAllForwardings=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), f"{user}@{host}",
        f"sed -n '{start},$p' /workspace/status.log 2>/dev/null || true",
    ], check=False)
    return p.stdout.splitlines() if p.returncode == 0 else []


def stop_remote(user: str, host: str, port: int) -> None:
    cmd = '''printf '%s | CANCEL_REQUESTED\\n' "$(date '+%H:%M:%S')" >> /workspace/status.log
if [ -s /workspace/run.pid ]; then
  pid=$(cat /workspace/run.pid)
  kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  sleep 8
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
fi'''
    run(["ssh", "-q", "-o", "LogLevel=ERROR", "-o", "ClearAllForwardings=yes", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), f"{user}@{host}", cmd], check=False)


def print_header(state: dict[str, Any]) -> None:
    offer = state.get("offer", {})
    print("=" * 58)
    print(f"Experiment : {state.get('run_script', '?')}")
    print(f"Instance   : {state.get('instance_id', '?')}")
    print(f"Offer      : {offer.get('id', '?')}")
    print(f"GPU        : {offer.get('gpu', '?')}")
    print(f"CPU        : {offer.get('cpu', '?')}")
    print(f"Rate       : ${float(offer.get('price') or 0.0):.3f}/h")
    print("=" * 58)


def delayed_kill_session(session: str) -> None:
    subprocess.Popen(
        ["sh", "-c", f"sleep 0.5; tmux kill-session -t {shlex.quote(session)} 2>/dev/null || true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def controller(state_path: Path) -> int:
    state = load_state(state_path)
    root = Path(state["project_root"]).resolve()
    session = state["session"]
    instance_id = int(state["instance_id"])
    run_script = state["run_script"]
    offer = state.get("offer", {})
    started = float(state.get("started", time.time()))
    poll = int(state.get("poll", 5))
    startup_timeout = int(state.get("startup_timeout", 600))
    sync_every = int(state.get("sync_every", 3600))
    next_sync = time.time() + sync_every

    print_header(state)
    log("TRACKING", f"active state: {state_path}")
    try:
        user, host, port = wait_for_ssh(instance_id, startup_timeout)
    except Exception as e:
        log("SSH_ERROR", str(e))
        return abort_instance(root, state_path, instance_id, session, run_script,
                              offer, started,
                              f"SSH never came up within {startup_timeout}s",
                              keep=bool(state.get("keep")))

    log("SSH_READY", f"ssh -p {port} {user}@{host}")
    ensure_monitor_windows(session, user, host, port)

    next_line = 1
    poll_errors = 0
    # Watchdog: the campaign must reach RUNNING (worker wrote /workspace/READY)
    # within this window, or the box is destroyed. Covers a wedged clone, a pip
    # install that hangs, a host that boots but never runs anything.
    startup_deadline = int(state.get("startup_deadline", 1800))
    boot_start = time.time()
    ever_running = False
    result: str | None = None
    cancelled = False
    while result is None:
        try:
            lines = status_lines(user, host, port, next_line)
            poll_errors = 0
            for line in lines:
                print(f"[remote] {line}", flush=True)
            next_line += len(lines)
            current = remote_state(user, host, port)
            if current in {"SUCCESS", "FAILED", "RUNNING"}:
                ever_running = True
            if current in {"SUCCESS", "FAILED"}:
                result = current
                break
            if (not ever_running and startup_deadline
                    and time.time() - boot_start > startup_deadline):
                return abort_instance(
                    root, state_path, instance_id, session, run_script, offer,
                    started,
                    f"campaign never started within {startup_deadline}s "
                    f"(last remote state: {current})",
                    keep=bool(state.get("keep")))
            # Mid-campaign sync: a long campaign should not risk losing every
            # finished run if the instance dies. Never fatal, never destroys.
            if sync_every and time.time() >= next_sync:
                try:
                    sync_instance(instance_id, root, run_name(run_script))
                    log("SYNC_PARTIAL", f"outputs+logs pulled at {int(time.time()-started)}s")
                except Exception as e:
                    log("SYNC_PARTIAL_FAILED", f"{e} (will retry)")
                next_sync = time.time() + sync_every
            time.sleep(poll)
        except KeyboardInterrupt:
            print("\nRemote experiment is still running.")
            print("  [c] continue tracking")
            print("  [q] quit local tracking; remote keeps running")
            print("  [s] stop remote run, sync, and destroy")
            while True:
                choice = input("Choice: ").strip().lower()
                if choice in {"c", "continue"}:
                    print()
                    break
                if choice in {"q", "quit"}:
                    log("DETACHING", "remote run continues; use --resume or --resume-all later")
                    delayed_kill_session(session)
                    return 0
                if choice in {"s", "stop"}:
                    log("STOPPING", "sending TERM to remote campaign")
                    stop_remote(user, host, port)
                    cancelled = True
                    deadline = time.time() + 45
                    while time.time() < deadline:
                        current = remote_state(user, host, port)
                        if current in {"SUCCESS", "FAILED"}:
                            result = current
                            break
                        time.sleep(2)
                    if result is None:
                        result = "FAILED"
                    break
                print("Enter c, q, or s.")
        except Exception as e:
            # Transient SSH/network errors are normal; a vanished instance is
            # not. After a few consecutive failures, ask Vast whether the box
            # still exists -- otherwise the controller polls a dead host and
            # "tracks" a run that can never finish.
            poll_errors += 1
            log("POLL_ERROR", f"{e} ({poll_errors})")
            if poll_errors >= 3:
                alive = instance_alive(instance_id)
                if alive is False:
                    log("VANISHED", f"instance {instance_id} no longer exists on Vast "
                                    "(destroyed externally, expired, or host offline)")
                    if not load_state(state_path).get("history_logged"):
                        append_history(root, "vanished", run_script, offer,
                                       instance_id, time.time() - started)
                    state_path.unlink(missing_ok=True)
                    log("CLEANED", "active state removed; nothing left to sync")
                    delayed_kill_session(session)
                    return 1
                log("ALIVE", f"instance {instance_id} still listed; retrying")
            time.sleep(poll)

    final_lines = status_lines(user, host, port, next_line)
    for line in final_lines:
        print(f"[remote] {line}", flush=True)

    log("FINISHED", "cancelled" if cancelled else result.lower())
    log("SYNCING", "outputs + diagnostics")
    try:
        sync_instance(instance_id, root, run_name(run_script))
    except Exception as e:
        log("SYNC_FAILED", str(e))
        log("KEEPING", f"instance {instance_id} and active state retained")
        return 1

    runtime = time.time() - started
    history_status = "cancelled" if cancelled else ("success" if result == "SUCCESS" else "failed")
    state = load_state(state_path)
    if not state.get("history_logged"):
        append_history(root, history_status, run_script, offer, instance_id, runtime)
        state = update_state(state_path, history_logged=True, final_status=history_status, sync_complete=True)
    else:
        state = update_state(state_path, sync_complete=True)

    if state.get("keep"):
        log("KEEPING", f"instance {instance_id} (--keep)")
        state_path.unlink(missing_ok=True)
        return 0 if result == "SUCCESS" else 1

    log("DESTROYING", str(instance_id))
    if not destroy_verified(instance_id):
        log("DESTROY_ERR", f"instance {instance_id} is STILL LISTED after retries "
                           f"-- it may still be billing. Destroy it in the web UI "
                           f"or: vastai destroy instance {instance_id}")
        log("KEEPING", "active state retained so --resume-all can retry cleanup")
        return 1
    log("DESTROYED", f"instance {instance_id} confirmed gone")

    state_path.unlink(missing_ok=True)
    log("DONE", f"{history_status} | {duration_text(runtime)}")
    delayed_kill_session(session)
    return 0 if result == "SUCCESS" else 1


def onstart_script() -> str:
    return '''mkdir -p /workspace
touch /workspace/status.log
status() { printf '%s | %s\\n' "$(date '+%H:%M:%S')" "$*" >> /workspace/status.log; }
(
  set -Eeuo pipefail
  exec >>/workspace/onstart.log 2>&1
  trap 'code=$?; status "FAILED onstart exit=$code"; touch /workspace/RUN_FAILED; exit $code' ERR

  status "ONSTART"
  status "INSTALLING system tools"
  apt-get update -qq
  apt-get install -y -qq git procps util-linux fonts-dejavu-core
  apt-get install -y -qq btop || status "WARNING btop install failed"

  ASKPASS=/tmp/syncnet-git-askpass
  cat >"$ASKPASS" <<'ASKPASS_EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\\n' "x-access-token" ;;
  *Password*) printf '%s\\n' "$GITHUB_TOKEN" ;;
esac
ASKPASS_EOF
  chmod 700 "$ASKPASS"

  # Vast does not always expose -e vars to the onstart shell; /etc/environment
  # is where they land. Source it before using GITHUB_TOKEN.
  if [ -z "${GITHUB_TOKEN:-}" ] && [ -f /etc/environment ]; then
    set -a; . /etc/environment; set +a
    status "ENV sourced /etc/environment"
  fi
  # Log enough to diagnose a clone failure without ever printing the token.
  status "REPO url=${REPO_URL:-UNSET} branch=${REPO_BRANCH:-UNSET} token_chars=${#GITHUB_TOKEN}"
  if [ -z "${GITHUB_TOKEN:-}" ]; then
    status "FAILED GITHUB_TOKEN is empty in the onstart environment"
  fi

  status "CLONING $REPO_BRANCH"
  if ! GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \\
      git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR" 2>/tmp/clone.err; then
    status "FAILED clone: $(tr -d '\\r' </tmp/clone.err | tail -n 2 | tr '\\n' ' ')"
    exit 128
  fi
  rm -f "$ASKPASS"

  status "STARTING worker"
  bash "$WORKDIR/bash_scripts/vast_worker" "$RUN_SCRIPT"
) &
exec /opt/instance-tools/bin/entrypoint.sh
'''


def create_instance(args, offer, root: Path, github: str, wandb: str, repo: str, branch: str, image: str) -> tuple[int, Path]:
    session = session_name(args.run_script)
    env_values = {
        "REPO_URL": repo,
        "REPO_BRANCH": branch,
        "WORKDIR": REMOTE_WORKDIR,
        "RUN_SCRIPT": args.run_script,
        "NGPU": (str(args.num_gpus) if args.num_gpus and args.num_gpus > 0 else "auto"),
        "GITHUB_TOKEN": github,
        "WANDB_API_KEY": wandb,
    }
    env_arg = " ".join(f"-e {k}={v}" for k, v in env_values.items())
    log("CREATING", f"offer={offer['id']} {offer.get('gpu', '?')} / {offer.get('cpu', '?')}")
    # ON-DEMAND ONLY. Passing --price to `vastai create instance` turns the
    # rental into a bid (interruptible): a higher bidder then stops the
    # instance mid-run and the campaign dies. We never pass it.
    create_cmd = [
        "vastai", "create", "instance", str(offer["id"]),
        "--image", image, "--label", session, "--env", env_arg,
        "--onstart-cmd", onstart_script(), "--disk", str(args.disk),
        "--ssh", "--direct", "--cancel-unavail", "--raw",
    ]
    assert not any(f in create_cmd for f in ("--price", "--bid")), \
        "on-demand invariant violated: bid flag present"
    p = run(create_cmd)
    instance_id = parse_instance_id(p.stdout)
    log("INSTANCE", str(instance_id))
    path = active_path(root, args.run_script)
    atomic_json_write(path, {
        "version": 1,
        "project_root": str(root),
        "run_script": args.run_script,
        "run_name": run_name(args.run_script),
        "session": session,
        "instance_id": instance_id,
        "offer": offer,
        "started": time.time(),
        "poll": args.poll,
        "startup_timeout": args.startup_timeout,
        "sync_every": args.sync_every,
        "startup_deadline": args.startup_deadline,
        "keep": bool(args.keep),
        # everything the controller needs to rent a REPLACEMENT box by itself
        "relaunch": {
            "retries_left": int(args.auto_retry),
            "max_ppg": float(args.max_price_per_gpu),
            "image": image, "disk": int(args.disk),
            "num_gpus": args.num_gpus, "min_cpus_per_gpu": args.min_cpus_per_gpu,
            "min_cuda": args.min_cuda, "verified_only": bool(args.verified_only),
            "repo": repo, "branch": branch,
            "gpu": list(args.gpu) or None, "min_reliability": args.min_reliability,
            "min_duration": args.min_duration,
        },
        "history_logged": False,
        "sync_complete": False,
    })
    return instance_id, path


def controller_command(state_path: Path) -> str:
    return shlex.join([sys.executable, str(Path(__file__).resolve()), "--_track-state", str(state_path)])


def ensure_tracking_session(state_path: Path) -> str:
    state = load_state(state_path)
    session = state["session"]
    cmd = controller_command(state_path)
    if not tmux_exists(session):
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "run", cmd], check=True)
        return "started"
    windows = tmux_windows(session)
    if "run" in windows:
        return "already-tracking"
    subprocess.run(["tmux", "new-window", "-d", "-t", session, "-n", "run", cmd], check=True)
    return "restarted-controller"


def cleanup_dead_state(root: Path, path: Path, state: dict) -> None:
    """Drop local tracking for an instance that no longer exists on Vast:
    kill the tmux session, log one history line, remove the active state."""
    iid = int(state["instance_id"])
    session = state.get("session")
    if session and shutil.which("tmux"):
        # never let a missing/misbehaving tmux abort the cleanup: the state
        # file removal below is the part that matters.
        try:
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        except Exception as e:
            print(f"(could not kill tmux session {session}: {e})")
    if not state.get("history_logged"):
        append_history(root, "vanished", state.get("run_script", "?"),
                       state.get("offer", {}), iid,
                       time.time() - float(state.get("started", time.time())))
    path.unlink(missing_ok=True)
    print(f"Instance {iid} no longer exists on Vast; removed tracking state"
          + (f" and session {session}." if session else "."))


def instance_alive(instance_id: int) -> bool | None:
    """True/False if we could ask Vast; None if the query itself failed
    (network blip -- callers must not treat that as 'gone')."""
    ids = active_instance_ids()
    if ids is None:
        return None
    return instance_id in ids


def destroy_verified(instance_id: int, attempts: int = 3) -> bool:
    """Destroy and CONFIRM it is gone. `vastai destroy` can return success
    while the instance lingers, which is how a box keeps billing after the
    controller thinks it is finished. Every step is logged: a silent failure
    here costs real money, so nothing is allowed to be swallowed."""
    for attempt in range(1, attempts + 1):
        try:
            # `vastai destroy instance` asks "[y/N]". In a non-interactive
            # controller that read hits EOF and is treated as N -- the box
            # survives while we log DESTROYING and move on. Answer it.
            p = subprocess.run(
                ["vastai", "destroy", "instance", str(instance_id)],
                text=True, capture_output=True, input="y\n", timeout=120,
            )
            out = ((p.stdout or "") + (p.stderr or "")).strip().replace("\n", " ")
            log("DESTROY_CMD", f"attempt {attempt}: rc={p.returncode} {out[:160]}")
        except Exception as e:
            log("DESTROY_CMD_ERROR", f"attempt {attempt}: {e}")
        time.sleep(4)
        try:
            alive = instance_alive(instance_id)
        except Exception as e:
            log("DESTROY_CHECK_ERROR", str(e))
            alive = None
        if alive is False:
            return True
        if alive is None:
            log("DESTROY_CHECK", "could not query Vast; retrying")
        else:
            log("DESTROY_RETRY", f"instance {instance_id} still listed (attempt {attempt})")
        time.sleep(4)
    try:
        return instance_alive(instance_id) is False
    except Exception:
        return False


def active_instance_ids() -> set[int] | None:
    p = run(["vastai", "show", "instances", "--raw"], check=False)
    if p.returncode != 0:
        return None
    data = parse_machine_output(p.stdout)
    items: list[dict[str, Any]] = []
    if isinstance(data, list):
        items = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        for key in ("instances", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                items = [x for x in value if isinstance(x, dict)]
                break
        if not items and ("id" in data or "contract_id" in data):
            items = [data]
    ids: set[int] = set()
    for item in items:
        raw = item.get("id", item.get("contract_id"))
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            pass
    return ids


def blacklist_path(root: Path) -> Path:
    return root / ".vast" / "blacklist.json"


def load_blacklist(root: Path) -> dict:
    p = blacklist_path(root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def blacklist_machine(root: Path, offer: dict, reason: str) -> None:
    """Remember hosts that failed to start. A machine whose container runtime
    is broken will fail the same way on every offer it lists, so blacklisting
    the MACHINE (not the offer) is what actually helps."""
    mid = offer.get("machine_id")
    if mid is None:
        return
    bl = load_blacklist(root)
    entry = bl.get(str(mid), {"failures": 0})
    entry["failures"] = entry.get("failures", 0) + 1
    entry["last_reason"] = reason[:200]
    entry["last_seen"] = time.strftime("%Y-%m-%d %H:%M")
    bl[str(mid)] = entry
    p = blacklist_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bl, indent=1, sort_keys=True))
    log("BLACKLISTED", f"machine {mid} ({entry['failures']}x): {reason[:80]}")


def blacklisted_machines(root: Path, min_failures: int = 1) -> set:
    out = set()
    for mid, e in load_blacklist(root).items():
        if e.get("failures", 0) >= min_failures:
            try:
                out.add(int(mid))
            except ValueError:
                out.add(mid)
    return out


def pick_offer_ladder(root: Path, base_kwargs: dict, max_ppg: float,
                      ladder=("A", "B", "C")) -> dict | None:
    """Cheapest box in the best tier available: try A within the price cap,
    then B, then C. Blacklisted machines are excluded at every step."""
    bl = blacklisted_machines(root)
    for tier in ladder:
        rows = search_offers(**base_kwargs, allowed_tiers={tier},
                             max_price_per_gpu=max_ppg, blacklist=bl)
        if rows:
            rows.sort(key=lambda r: (r["price_per_gpu"], -(r["st_score"] or 0)))
            pick = rows[0]
            log("AUTO_PICK", f"tier {tier}: offer {pick['id']} machine "
                             f"{pick.get('machine_id')} "
                             f"${pick['price_per_gpu']:.3f}/GPU/h "
                             f"{pick['num_gpus']}x {pick['gpu']}")
            return pick
        log("AUTO_PICK", f"no tier-{tier} offer under ${max_ppg:.2f}/GPU/h")
    return None


def relaunch_after_abort(root: Path, state_path: Path, state: dict,
                         github: str, wandb_key: str) -> int | None:
    """After an aborted boot, rent a replacement: cheapest A-tier under the
    price cap, else B, else C, never a blacklisted machine. Returns the new
    instance id, or None if no retries remain / nothing suitable exists."""
    rl = dict(state.get("relaunch") or {})
    if int(rl.get("retries_left", 0)) <= 0:
        return None
    base = dict(gpus=rl.get("gpu"), min_reliability=rl.get("min_reliability", 0.99),
                min_cpus=0, min_disk=rl.get("disk", 25),
                min_duration=rl.get("min_duration", 1.0), limit=1000,
                exact_gpus=(rl.get("num_gpus") or None),
                min_cpus_per_gpu=rl.get("min_cpus_per_gpu", 8.0),
                min_cuda=rl.get("min_cuda"),
                verified_only=bool(rl.get("verified_only")))
    offer = pick_offer_ladder(root, base, float(rl.get("max_ppg", 0.70)))
    if offer is None:
        log("RETRY_FAILED", "no eligible A/B/C offer under the price cap")
        return None
    rl["retries_left"] = int(rl["retries_left"]) - 1
    ns = SimpleNamespace(run_script=state["run_script"], disk=rl.get("disk", 25),
                         num_gpus=rl.get("num_gpus"), poll=int(state.get("poll", 5)),
                         startup_timeout=int(state.get("startup_timeout", 600)),
                         sync_every=int(state.get("sync_every", 3600)),
                         startup_deadline=int(state.get("startup_deadline", 1800)),
                         keep=bool(state.get("keep")), auto_retry=rl["retries_left"],
                         max_price_per_gpu=rl.get("max_ppg", 0.70),
                         min_cpus_per_gpu=rl.get("min_cpus_per_gpu", 8.0),
                         min_cuda=rl.get("min_cuda"), gpu=rl.get("gpu") or [],
                         min_reliability=rl.get("min_reliability", 0.99),
                         min_duration=rl.get("min_duration", 1.0))
    iid, _ = create_instance(ns, offer, root, github, wandb_key,
                             rl.get("repo"), rl.get("branch"), rl.get("image"))
    log("RELAUNCHED", f"instance {iid} (retries left: {rl['retries_left']})")
    return iid


def abort_instance(root: Path, state_path: Path, instance_id: int, session: str,
                   run_script: str, offer: dict, started: float, reason: str,
                   keep: bool = False) -> int:
    """Give up on a booting instance: pull whatever diagnostics exist, destroy
    it, and clear local tracking. An instance that never starts still bills."""
    log("ABORTING", reason)
    blacklist_machine(root, offer, reason)
    try:
        sync_instance(instance_id, root, run_name(run_script))
        log("SYNCED", "diagnostics pulled before abort")
    except Exception as e:
        log("SYNC_SKIPPED", f"{e}")
    # The host-side boot log lives on Vast, not on the instance filesystem, so
    # a failed container start leaves nothing to rsync. Save it separately.
    try:
        text = instance_logs(instance_id, tail=400)
        if text.strip():
            d = root / ".vast" / "remote_logs" / run_name(run_script) / str(instance_id)
            d.mkdir(parents=True, exist_ok=True)
            (d / "instance_boot.log").write_text(text)
            log("SAVED", f"instance boot log -> {d / 'instance_boot.log'}")
    except Exception as e:
        log("LOGS_SKIPPED", f"{e}")
    if keep:
        log("KEEPING", f"instance {instance_id} retained (--keep)")
        return 1
    if destroy_verified(instance_id):
        log("DESTROYED", f"instance {instance_id} confirmed gone")
    else:
        log("DESTROY_ERR", f"could NOT confirm destruction of {instance_id} -- "
                           f"check the web UI; it may still bill")
    try:
        st = load_state(state_path)
    except Exception:
        st = {"instance_id": instance_id, "session": session, "run_script": run_script}
    if not st.get("history_logged"):
        append_history(root, "aborted", run_script, offer, instance_id,
                       time.time() - started)
    st["history_logged"] = True
    # Try a replacement box before giving up entirely.
    try:
        github = os.environ.get("GITHUB_TOKEN", "")
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if github and wandb_key:
            new_id = relaunch_after_abort(root, state_path, st, github, wandb_key)
            if new_id:
                return 0   # a fresh controller now tracks the replacement
    except Exception as e:
        log("RETRY_ERROR", str(e))
    cleanup_dead_state(root, state_path, st)
    return 1


def destroy_campaign(root: Path, name: str, force: bool = False, skip_sync: bool = False) -> int:
    """Tear a campaign down properly: pull results, destroy the instance, then
    remove the local tracking state and tmux session. Refuses to destroy if
    the sync fails (use --force to override) so results are never lost."""
    path = resolve_resume_path(root, name)
    state = load_state(path)
    iid = int(state["instance_id"])
    run_script = state.get("run_script", name)

    alive = instance_alive(iid)
    if alive is False:
        print(f"Instance {iid} is already gone on Vast.")
        cleanup_dead_state(root, path, state)
        return 0

    if not skip_sync:
        try:
            sync_instance(iid, root, run_name(run_script))
            print(f"Synced outputs + logs for {run_script}.")
        except Exception as e:
            print(f"SYNC FAILED: {e}")
            if not force:
                print("Refusing to destroy -- results would be lost. "
                      "Re-run with --force to destroy anyway, or fix the sync first.")
                return 1
            print("--force given: destroying despite the failed sync.")

    if not destroy_verified(iid):
        print(f"Could NOT confirm destruction of {iid}; it may still be billing. "
              f"Check: vastai show instances")
        return 1
    print(f"Instance {iid} destroyed and confirmed gone.")
    cleanup_dead_state(root, path, state)
    return 0


def resume_all(root: Path) -> None:
    paths = sorted(active_dir(root).glob("*.json"))
    if not paths:
        print("No tracked Vast runs to resume.")
        return
    active_ids = active_instance_ids()
    started = stale = already = 0
    for path in paths:
        try:
            state = load_state(path)
            iid = int(state["instance_id"])
            script = state["run_script"]
        except Exception as e:
            print(f"Skipping unreadable state {path}: {e}")
            continue
        if active_ids is not None and iid not in active_ids:
            print(f"Stale: {script} -> instance {iid} is no longer active.")
            cleanup_dead_state(root, path, state)
            stale += 1
            continue
        result = ensure_tracking_session(path)
        if result == "already-tracking":
            print(f"Already tracking: {script} -> {state['session']}")
            already += 1
        else:
            print(f"Resumed: {script} -> {state['session']} (instance {iid})")
            started += 1
    print(f"\nResume summary: {started} resumed, {already} already tracked, {stale} stale state(s) removed.")
    if started:
        print("Use 'tmux ls' to see the restored run sessions.")


def resolve_resume_path(root: Path, target: str) -> Path:
    matches = []
    for path in sorted(active_dir(root).glob("*.json")):
        try:
            state = load_state(path)
        except Exception:
            continue
        if target in {state.get("run_name"), state.get("run_script"), state.get("session"), path.stem}:
            matches.append(path)
    if not matches:
        tracked = [p.stem for p in sorted(active_dir(root).glob("*.json"))]
        hist = root / ".vast" / "run_history.log"
        last = ""
        if hist.is_file():
            lines = [l for l in hist.read_text().splitlines() if l.strip()]
            if lines:
                last = f"\n  last finished run: {lines[-1][:120]}"
        raise SystemExit(
            f"No tracked active run matches {target!r}.\n"
            f"  currently tracked: {tracked or 'none -- every run has finished and cleaned up'}"
            f"{last}")
    if len(matches) > 1:
        raise SystemExit(f"Multiple active runs match {target!r}: {[p.stem for p in matches]}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch, track, resume, sync, and clean up disposable Vast experiment workers.")
    ap.add_argument("offer", nargs="?", help="Vast OFFER_ID, or 'auto'")
    ap.add_argument("run_script", nargs="?", help="Repo-relative Bash script, e.g. bash_scripts/explore1")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--repo")
    ap.add_argument("--branch")
    ap.add_argument("--image")
    # Measured footprint per box: base image ~5-8 GB + pip ~2 GB + the one
    # dataset this campaign uses (sqoop rhs ~0.8 GB compressed, SoC <0.5 GB)
    # + checkpoints/wandb cache ~2-4 GB. 25 GB leaves roughly 2x headroom and
    # keeps far more offers eligible than the old 200 GB floor.
    ap.add_argument("--disk", type=int, default=25,
                    help="Instance disk in GB; also the marketplace floor "
                         "(offers with less available disk are excluded).")
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="Rent a box with EXACTLY this many GPUs. Default: any "
                         "count -- 1/2/4/8-GPU offers are all shown, and the "
                         "campaign runs one independent experiment per GPU on "
                         "whatever you pick, from a shared queue.")
    ap.add_argument("--min-cpus-per-gpu", type=float, default=8.0)
    ap.add_argument("--force", action="store_true",
                    help="With --destroy: destroy even if the sync fails.")
    ap.add_argument("--no-sync", action="store_true",
                    help="With --destroy: skip the sync entirely (results are lost).")
    ap.add_argument("--auto-retry", type=int, default=2,
                    help="On a failed boot: blacklist the machine and rent a "
                         "replacement, up to this many times (0 disables).")
    ap.add_argument("--max-price-per-gpu", type=float, default=0.70,
                    help="Price cap used when auto-picking a replacement box.")
    ap.add_argument("--verified-only", action="store_true",
                    help="Only rent Vast-verified hosts (fewer boot failures, "
                         "slightly higher prices).")
    ap.add_argument("--min-cuda", type=float, default=None,
                    help="Minimum host CUDA version; match your image (e.g. 13.0).")
    ap.add_argument("--startup-deadline", type=int, default=1800,
                    help="Destroy the instance if the campaign has not started "
                         "within this many seconds (default 1800 = 30 min; "
                         "0 disables). --keep overrides the destroy.")
    ap.add_argument("--sync-every", type=int, default=3600,
                    help="Seconds between mid-campaign output syncs (0 disables).")
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--poll", type=int, default=5)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--gpu", action="append", default=[])
    ap.add_argument("--max-price", type=float)
    ap.add_argument("--min-reliability", type=float, default=0.99)
    ap.add_argument("--min-cpus", type=float, default=8)
    ap.add_argument("--min-duration", type=float, default=1.0)
    ap.add_argument("--tiers", default="A,B,C")
    ap.add_argument("--search-limit", type=int, default=1000)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--resume", metavar="RUN", help="Resume tracking one active run by script/run/session name")
    ap.add_argument("--destroy", metavar="NAME|ID",
                    help="Destroy an instance (campaign name or raw id), verify it is "
                         "gone, and clear local tracking.")
    ap.add_argument("--version", action="store_true",
                    help="Print the runner version + enabled features and exit.")
    ap.add_argument("--resume-all", action="store_true", help="Restore tracking for every .vast/active run")
    ap.add_argument("--_track-state", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.version:
        print(f"vast_run.py {RUNNER_VERSION}")
        for f in RUNNER_FEATURES:
            print(f"  + {f}")
        return


    root = Path(args.project_root).resolve()
    for tool in ("vastai", "ssh", "rsync", "tmux"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is required locally")

    if args._track_state:
        raise SystemExit(controller(Path(args._track_state).resolve()))
    if args.resume_all:
        resume_all(root)
        return
    if args.destroy:
        # Sync results, destroy, verify, then clear local tracking.
        # Accepts a campaign name (state file is cleaned up too) or a raw id.
        target = args.destroy
        try:
            iid = int(target)
        except ValueError:
            raise SystemExit(destroy_campaign(root, target,
                                              force=args.force,
                                              skip_sync=args.no_sync))
        print(f"Destroying instance {iid} ...")
        okgone = destroy_verified(iid)
        print("Destroyed and confirmed gone." if okgone else
              f"COULD NOT CONFIRM destruction of {iid} -- check the web UI; it may still bill.")
        # clear any local state that referenced this id
        for cand in sorted(active_dir(root).glob("*.json")):
            try:
                st = load_state(cand)
            except Exception:
                continue
            if int(st.get("instance_id", -1)) == iid:
                cleanup_dead_state(root, cand, st)
        raise SystemExit(0 if okgone else 1)

    if args.resume:
        path = resolve_resume_path(root, args.resume)
        state = load_state(path)
        # A tmux session existing does not mean the run does. Check Vast first,
        # or --resume happily reports "already-tracking" for a destroyed box.
        if instance_alive(int(state["instance_id"])) is False:
            cleanup_dead_state(root, path, state)
            return
        result = ensure_tracking_session(path)
        state = load_state(path)
        print(f"{result}: {state['run_script']} -> {state['session']} (instance {state['instance_id']})")
        return
    if not args.offer or not args.run_script:
        raise SystemExit("Normal launch usage: vast_run.py <OFFER_ID|auto> <bash_script>")

    script = Path(args.run_script)
    if script.is_absolute() or not (root / script).is_file():
        raise SystemExit(f"Run script not found locally: {root / script}")
    if not (root / "bash_scripts" / "vast_worker").is_file():
        raise SystemExit("Missing bash_scripts/vast_worker")

    session = session_name(args.run_script)
    state_path = active_path(root, args.run_script)
    if state_path.exists():
        state = load_state(state_path)
        raise SystemExit(
            f"{args.run_script} already has a tracked Vast instance {state.get('instance_id')}.\n"
            f"Use: {Path(__file__).name} --resume {run_name(args.run_script)}\n"
            f"or:  {Path(__file__).name} --resume-all"
        )
    if tmux_exists(session):
        raise SystemExit(f"{session} already exists. Attach with: tmux attach -t {session}")

    load_dotenv(root / ".env")
    github = require_env("GITHUB_TOKEN")
    wandb = require_env("WANDB_API_KEY")
    repo = args.repo or os.environ.get("VAST_REPO_URL", DEFAULT_REPO)
    branch = args.branch or os.environ.get("VAST_REPO_BRANCH", DEFAULT_BRANCH)
    image = args.image or os.environ.get("VAST_IMAGE", DEFAULT_IMAGE)

    if already_succeeded(root, args.run_script) and not args.rerun:
        if input(f"{args.run_script} already succeeded before. Run again? [y/N] ").strip().lower() not in {"y", "yes"}:
            raise SystemExit("Cancelled.")

    git_preflight(root, branch, args.run_script)
    preflight_repo(repo, branch, github)
    auto_mode = args.offer.lower() == "auto"
    while True:
        if auto_mode:
            offer = choose_auto(args)
        else:
            try:
                offer_id = int(args.offer)
            except ValueError:
                raise SystemExit("offer must be an OFFER_ID or 'auto'")
            offer = lookup_offer(offer_id) or {
                "id": offer_id, "gpu": "?", "cpu": "?", "price": 0.0, "st_pct": None,
                "tier": "?", "ghz": 0.0, "vcpus": 0.0, "rel": 0.0, "pcie": "?",
                "pcie_bw": 0.0, "inet_down": 0.0, "duration": 0.0, "loc": "?",
            }
            print()
            print_candidate(offer)
        try:
            instance_id, state_path = create_instance(args, offer, root, github, wandb, repo, branch, image)
            break
        except Exception as e:
            print(f"\nInstance creation failed: {e}")
            if not auto_mode:
                raise SystemExit(1)
            if input("Refresh the candidate list and try again? [Y/q] ").strip().lower() in {"q", "quit"}:
                raise SystemExit("Cancelled.")

    result = ensure_tracking_session(state_path)
    print(f"\nLaunched {args.run_script} on instance {instance_id}.")
    print(f"Tracking: {session} ({result})")
    print(f"Attach:   tmux attach -t {session}")
    print("Windows:  run | ssh | btop | gpu")
    print("This command is finished; the Vast run continues independently.")
    print("After a reboot: python scripts/vast/vast_run.py --resume-all")


if __name__ == "__main__":
    main()
