"""Structure audit: the locked v16 touch/new/delete tree, checked against
the filesystem. NEW paths must exist, DELETE paths must be gone, TOUCH
paths must exist AND contain the marker string of their specific change,
and a handful of KEEP spot-checks confirm the untouched surface is intact.
Run after any restructuring; a red line here means the tree and the repo
have drifted."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEW = [
    # --- campaigns, tooling and hybrid cells (added during the v16 campaign) ---
    'bash_scripts/_campaign_lib.sh',
    'bash_scripts/vast_worker',
    'bash_scripts/run_smoke_test',
    'bash_scripts/run_sort_of_clevr',
    'bash_scripts/run_sqoop_rhs18',
    'bash_scripts/run_sqoop_rhs01',
    'bash_scripts/run_sqoop_rhs02',
    'bash_scripts/run_sqoop_rhs04',
    'bash_scripts/run_sqoop_rhs08',
    'bash_scripts/run_sqoop_rhs35',
    'bash_scripts/run_sqoop_plateau_probe',
    'bash_scripts/run_sqoop_optim_probe',
    'bash_scripts/run_sqoop_rhs18_bestofboth',
    'bash_scripts/run_sqoop_rhs01_gate_axes',
    'bash_scripts/run_soc_readout_prior',
    'scripts/diagnose_signal.py',
    'scripts/run_interventions.py',
    'scripts/snapshot_wandb.py',
    'scripts/vast/vast_run.py',
    'scripts/vast/vast_find.py',
    'scripts/vast/vast_sync_outputs.py',
    'conf/experiment/sqoop/sync/hybrids/identity_partition.yaml',
    'conf/experiment/sqoop/sync/hybrids/identity_partition_cnnenc.yaml',
    'conf/experiment/sqoop/sync/hybrids/canonical_partition.yaml',
    'conf/experiment/sqoop/sync/hybrids/gated_pooled.yaml',
    'conf/experiment/sqoop/sync/hybrids/gated_noprior.yaml',
    'conf/experiment/sort_of_clevr/sync/hybrids/identity_spatial_noprior.yaml',
    'conf/experiment/sort_of_clevr/sync/hybrids/identity_partition.yaml',
    'conf/experiment/sort_of_clevr/sync/hybrids/gated_pooled.yaml',
    'src/models/sync/components/__init__.py',
    'src/models/sync/components/binding.py',
    'src/models/sync/components/identity.py',
    'src/models/sync/components/medium.py',
    'src/models/sync/components/dynamics.py',
    'src/models/sync/components/readout.py',
    'src/models/sync/components/interventions.py',
    'src/models/sync/field.py',
    'src/models/sync/conditioning.py',
    'src/models/sync/busnet.py',
    'src/models/sync/identity_busnet.py',
    'src/models/sync/token_busnet.py',
    'src/models/sync/gated.py',
    'src/analysis/probes.py',
    'verify/verify_components.py',
    'verify/verify_reference.py',
    'verify/verify_segregation.py',
    'verify/verify_gates.py',
    'verify/verify_training_oracle.py',
    'verify/fixtures/canonical_golden.pt',
    'verify/fixtures/oracle_canonical_curve.pt',
    'conf/model/sort_of_clevr/identity_busnet.yaml',
    'conf/model/sqoop/identity_busnet.yaml',
    'conf/model/sort_of_clevr/gated.yaml',
    'conf/model/sqoop/gated.yaml',
    'conf/model/sort_of_clevr/token_busnet.yaml',
    'conf/model/sqoop/token_busnet.yaml',
    'conf/experiment/sort_of_clevr/_protocol.yaml',
    'conf/experiment/sqoop/_protocol.yaml',
    'conf/experiment/sort_of_clevr/sync/_family.yaml',
    'conf/experiment/sqoop/sync/_family.yaml',
    'conf/experiment/sort_of_clevr/baselines/_family.yaml',
    'conf/experiment/sqoop/baselines/_family.yaml',
    'conf/experiment/sort_of_clevr/sync/thesis/canonical.yaml',
    'conf/experiment/sort_of_clevr/baselines/matched_ch/_family.yaml',
    'conf/experiment/sort_of_clevr/baselines/matched_ch/shared_workspace.yaml',
    'conf/experiment/sqoop/baselines/matched_ch/_family.yaml',
    'conf/experiment/sqoop/baselines/matched_ch/shared_workspace.yaml',
    'conf/experiment/sqoop/baselines/matched_ch/transformer.yaml',
    'conf/experiment/sort_of_clevr/matched/_family.yaml',
    'conf/experiment/sort_of_clevr/matched/film_fieldenc.yaml',
    'conf/experiment/sort_of_clevr/matched/conv_fieldenc.yaml',
    'conf/experiment/sort_of_clevr/matched/busnet_cnnenc.yaml',
    'conf/experiment/sort_of_clevr/sync/thesis/identity.yaml',
    'conf/experiment/sort_of_clevr/sync/thesis/gated.yaml',
    'conf/experiment/sort_of_clevr/sync/thesis/tokens.yaml',
    'conf/experiment/sqoop/sync/thesis/canonical.yaml',
    'conf/experiment/sqoop/sync/thesis/identity.yaml',
    'conf/experiment/sqoop/sync/thesis/gated.yaml',
    'conf/experiment/sqoop/sync/thesis/tokens_stim.yaml',
    'conf/experiment/sqoop/sync/thesis/tokens_static.yaml',
    'conf/experiment/sort_of_clevr/sync/ablations/medium_silent.yaml',
    'conf/experiment/sort_of_clevr/sync/ablations/addresses_static.yaml',
    'conf/experiment/sort_of_clevr/sync/ablations/identity_cells_only.yaml',
    'conf/experiment/sort_of_clevr/sync/ablations/gated_attn.yaml',
    'conf/experiment/sqoop/sync/ablations/medium_lines_full.yaml',
    'conf/experiment/sqoop/sync/ablations/addresses_open.yaml',
    'conf/experiment/sqoop/sync/ablations/identity_anchors_only.yaml',
    'conf/experiment/sqoop/sync/ablations/gated_full.yaml',
    'conf/experiment/sort_of_clevr/baselines/thesis/relnet.yaml',
    'conf/experiment/sqoop/baselines/thesis/film.yaml',
    'conf/callbacks/sort_of_clevr/sync_v16.yaml',
]

TOUCH = {  # path -> marker of the specific change
    'src/datasets/soc/loader.py': "'scenes'",
    'src/datasets/sqoop/loader.py': "'scenes'",
    'src/datasets/sqoop/generator.py': 'SHAPE_TO_IDX',
    'conf/experiment/sqoop/archive/busnet/_base.yaml': 'weight_decay',
    'conf/experiment/sqoop/archive/busnet/stim.yaml': 'addresses',
    'src/models/sync/busnet.py': 'supported_callbacks',
    'src/analysis/interventions.py': 'anchor_shuffle',
    'src/analysis/sync_metrics.py': 'fixed_point_signature',
    'conf/README.md': 'v16',
    'conf/model/sort_of_clevr/busnet.yaml': 'addresses',
    'conf/model/sqoop/busnet.yaml': 'addresses',
    'src/models/__init__.py': "'gated'",
    'conf/task/sort_of_clevr.yaml': 'busnet',
    'conf/task/sqoop.yaml': 'busnet',
}

ARCHIVED = [
    'conf/experiment/sort_of_clevr/archive/sync_d',
    'conf/experiment/sort_of_clevr/archive/baselines',
    'conf/experiment/sqoop/archive/syncnet',
    'conf/experiment/sqoop/archive/diagnostics',
]

DELETE = [
    'src/models/sync/syncnet.py',
    'src/models/sync/phasebind.py',
    'src/models/sync/fieldsync.py',
    'src/models/sync/osc_field.py',
    'src/models/sync/common',
    'conf/model/sort_of_clevr/syncnet.yaml',
    'conf/model/sort_of_clevr/phasebind.yaml',
    'conf/model/sort_of_clevr/fieldsync.yaml',
    'conf/model/sort_of_clevr/osc_field.yaml',
    'conf/model/sqoop/syncnet.yaml',
]

KEEP = [
    'src/models/common/img_enc.py', 'src/models/baseline/conv.py',
    'src/models/baseline/relnet.py', 'src/training', 'src/core/registry.py',
    'scripts/main.py', 'src/analysis/sync_viz.py', 'src/analysis/t_variance.py',
    'src/datasets/sqoop/generator.py',
]

NOT_CONTAIN = {'src/models/__init__.py': ["'syncnet'", 'phasebind', 'fieldsync', 'osc_field']}


def main() -> int:
    bad = 0
    for p in NEW:
        ok = (ROOT / p).exists()
        bad += not ok
        print(f'  {"ok " if ok else "MISSING"} NEW     {p}')
    for p, marker in TOUCH.items():
        f = ROOT / p
        ok = f.exists() and marker in f.read_text()
        bad += not ok
        print(f'  {"ok " if ok else "FAIL"} TOUCH   {p}  [{marker}]')
    for p in DELETE:
        ok = not (ROOT / p).exists()
        bad += not ok
        print(f'  {"ok " if ok else "STILL PRESENT"} DELETE  {p}')
    for p in ARCHIVED:
        ok = (ROOT / p).exists()
        bad += not ok
        print(f'  {"ok " if ok else "MISSING"} ARCHIVE {p}')
    for p in KEEP:
        ok = (ROOT / p).exists()
        bad += not ok
        print(f'  {"ok " if ok else "MISSING"} KEEP    {p}')
    for p, markers in NOT_CONTAIN.items():
        text = (ROOT / p).read_text()
        for mk in markers:
            ok = mk not in text
            bad += not ok
            print(f'  {"ok " if ok else "FAIL"} PURGED  {p} has no {mk}')
    print(f'\n{"TREE VALID" if bad == 0 else f"{bad} DEVIATIONS"}: '
          f'{len(NEW)} new, {len(TOUCH)} touched, {len(DELETE)} deleted, {len(KEEP)} keep spot-checks')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
