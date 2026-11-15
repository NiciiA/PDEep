"""Scope new evaluation registries; reuse existing verified artifact synchronization."""
from pathlib import Path
from dataclasses import replace
from ..artifact_drive import PersistencePolicy
from ..unified_bayes.storage import ArtifactStore


def evaluation_store(root,policy=None,*,experiment,selection,comparison):
    policy=policy or PersistencePolicy();store=ArtifactStore(root,policy)
    if policy.drive_enabled:
        # Avoid overwriting the numerical-baseline or another checkpoint's registry.
        base=Path(policy.drive_root or policy.drive_mount)
        store.policy=replace(store.policy,drive_root=str(base/'G1_unified'/'evaluations-modes'/experiment/selection/comparison))
    return store


def archive_summary(store,report,plots,*,raw_indices):
    """Archive derived summaries/PNGs; raw rows already have case manifests."""
    import numpy as np
    import torch
    from ..core import digest
    from ..artifact_io import write_artifact,sha256
    summary={k:v for k,v in report.items() if k!='rows'}
    if 'inverse_six_method' in summary:summary['inverse_six_method']={k:v for k,v in summary['inverse_six_method'].items() if k!='rows'}
    payload={'summary':summary,'raw_case_indices':list(raw_indices),'figure_png':{Path(p).name+'-'+str(i):torch.from_numpy(np.frombuffer(Path(p).read_bytes(),dtype=np.uint8).copy()) for i,p in enumerate(plots)}}
    key=digest((summary,{str(p):sha256(p) for p in plots},list(raw_indices)))
    path=write_artifact(store.root/'derived',key,{'artifact_type':'diagnostic_bundle','architecture_id':'G1_MODES_COMPARISON','run_id':'G1-three-mode-derived-results'},payload,required=('summary','raw_case_indices','figure_png'),idempotent=True)
    return str(path),store(path)
