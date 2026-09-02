#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

REMOTE_WORKDIR = "/workspace/SyncNetProject"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(cmd, text=True, capture_output=True)
    if check and p.returncode:
        raise RuntimeError((p.stderr or p.stdout).strip())
    return p


def get_ssh(instance_id: int) -> tuple[str, str, int]:
    p = run(["vastai", "ssh-url", str(instance_id)])
    m = re.search(r"ssh://([^@\s]+)@([^:\s/]+):(\d+)", p.stdout)
    if not m:
        raise RuntimeError(f"Could not parse ssh-url output: {p.stdout!r}")
    return m.group(1), m.group(2), int(m.group(3))


def ssh_base(user: str, host: str, port: int) -> list[str]:
    return [
        "ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8",
        "-p", str(port), f"{user}@{host}",
    ]


def wait_for_ssh(instance_id: int, timeout: int = 300) -> tuple[str, str, int]:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            user, host, port = get_ssh(instance_id)
            p = run(ssh_base(user, host, port) + ["true"], check=False)
            if p.returncode == 0:
                return user, host, port
            last = (p.stderr or p.stdout).strip()
        except Exception as e:
            last = str(e)
        time.sleep(3)
    raise RuntimeError(f"Timed out waiting for SSH. Last error: {last}")


def remote_exists(user: str, host: str, port: int, path: str, directory: bool = False) -> bool:
    flag = "-d" if directory else "-f"
    return run(ssh_base(user, host, port) + [f"test {flag} {path}"], check=False).returncode == 0


def rsync_path(user: str, host: str, port: int, remote: str, local: Path, directory: bool = False) -> None:
    if directory:
        local.mkdir(parents=True, exist_ok=True)
    else:
        local.parent.mkdir(parents=True, exist_ok=True)

    transport = f"ssh -p {port} -o StrictHostKeyChecking=accept-new"
    source = f"{user}@{host}:{remote}"
    destination = str(local) + ("/" if directory else "")

    # Deliberately allow updates: interrupted rsyncs can finish correctly on resume.
    p = subprocess.run([
        "rsync", "-a", "--partial-dir=.rsync-partial",
        "--info=stats1", "-e", transport, source, destination,
    ])
    if p.returncode:
        raise RuntimeError(f"rsync failed for {remote} with exit code {p.returncode}")


def sync_instance(
    instance_id: int,
    project_root: Path,
    run_name: str,
    remote_workdir: str = REMOTE_WORKDIR,
    timeout: int = 300,
) -> None:
    for tool in ("vastai", "ssh", "rsync"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is required locally")

    user, host, port = wait_for_ssh(instance_id, timeout)

    remote_outputs = f"{remote_workdir.rstrip('/')}/outputs/"
    local_outputs = project_root / "outputs"
    if remote_exists(user, host, port, remote_outputs.rstrip("/"), directory=True):
        rsync_path(user, host, port, remote_outputs, local_outputs, directory=True)

    # Campaign logs (bash_scripts/logs/<campaign>/*.log) are the tee'd
    # per-run training logs. They are NOT under outputs/, so without this
    # they would be destroyed with the instance.
    remote_camp = f"{remote_workdir.rstrip('/')}/bash_scripts/logs/{run_name}/"
    local_camp = project_root / "bash_scripts" / "logs" / run_name
    if remote_exists(user, host, port, remote_camp.rstrip("/"), directory=True):
        rsync_path(user, host, port, remote_camp, local_camp, directory=True)

    diag_dir = project_root / ".vast" / "remote_logs" / run_name / str(instance_id)
    for remote in (
        "/workspace/status.log",
        "/workspace/onstart.log",
        "/workspace/bootstrap.log",
        "/workspace/run.log",
        "/workspace/verify.log",
    ):
        if remote_exists(user, host, port, remote):
            rsync_path(user, host, port, remote, diag_dir / Path(remote).name, directory=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync a Vast worker's outputs and infrastructure logs locally.")
    ap.add_argument("instance_id", type=int)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--run-name", default="manual")
    ap.add_argument("--remote-workdir", default=REMOTE_WORKDIR)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    sync_instance(
        args.instance_id,
        Path(args.project_root).resolve(),
        args.run_name,
        args.remote_workdir,
        args.timeout,
    )
    print("Sync complete.")


if __name__ == "__main__":
    main()
