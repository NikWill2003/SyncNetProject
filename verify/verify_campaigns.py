#!/usr/bin/env python3
"""Campaign audit: every experiment= path must EXIST as a yaml file.

Hydra treats the `experiment` group as optional, so a typo like
    experiment=sort_of_clevr/sync/thesis/canonicl
does NOT raise -- it silently composes the DEFAULT config and trains the
wrong model under a right-looking run name. compose()/build_model() cannot
catch that; only a file-existence check can.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    bad, checked, campaigns = [], 0, 0
    for script in sorted((ROOT / "bash_scripts").glob("run_*")):
        campaigns += 1
        text = script.read_text()
        for name, args in re.findall(r'"(\S+)\s+(task=\S+.*?)"', text):
            for key, sub in (("experiment", "experiment"), ("model", "model")):
                m = re.search(rf"\b{key}=(\S+)", args)
                if not m:
                    continue
                value = m.group(1)
                if value.startswith("NO_SUCH") or "$" in value:
                    continue
                checked += 1
                if not (ROOT / "conf" / sub / f"{value}.yaml").is_file():
                    bad.append(f"{script.name}: {name}: {key}={value} -> "
                               f"conf/{sub}/{value}.yaml MISSING")
    for b in bad:
        print("  BAD", b)
    print(f"{campaigns} campaigns, {checked} group references checked: "
          f"{len(bad)} missing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
