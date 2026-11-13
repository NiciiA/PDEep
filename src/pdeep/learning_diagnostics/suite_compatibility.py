"""Explicit checked continuation of pre-suite fixed/v2 diagnostics.

Only orchestration, optional v3 sampling, diagnostic time grids and per-quantity
logging changed. Existing v2 streams and numerical losses have regression tests;
models, physical data, reference solvers and shared optimizer are hash guarded.
"""
import json
from pathlib import Path
from ..core import digest,require
from ..artifact_io import sha256


def approved_diagnostic_predecessor(config):
    root=Path(__file__).resolve().parents[3]
    wave=root/'certifications/G1_smooth_v1/portable/wave_kinematic_execution.json'
    if wave.is_file():
        proof=json.loads(wave.read_text())
        require(digest({k:v for k,v in proof.items() if k!='identity'})==proof['identity'],'wave compatibility proof checksum mismatch')
        from .engine import diagnostic_source
        if diagnostic_source()==proof['new_diagnostic_source'] and config.diagnostic_source_id in proof['approved_predecessors']:
            # Optional new term is OFF for every predecessor. Never reinterpret
            # an old weighted/staged objective or its query/resume state.
            if getattr(config,'kinematic_weight',0.)!=0 or config.physics_schedule not in {'constant','staged_300_700'}:return False
            require(all(sha256(root/p)==sha for p,sha in proof['unchanged_files'].items()),'wave compatibility requires unchanged model/physics/reference/shared optimizer')
            return True
    path=root/'certifications/G1_smooth_v1/portable/optimization_diagnostics_execution.json'
    if not path.exists():path=root/'certifications/G1_smooth_v1/portable/diagnostic_suite_execution.json'
    if not path.exists():return False
    proof=json.loads(path.read_text())
    require(digest({k:v for k,v in proof.items() if k!='identity'})==proof['identity'],'suite compatibility proof checksum mismatch')
    from .engine import diagnostic_source
    if config.diagnostic_source_id not in proof['approved_predecessors'] or diagnostic_source()!=proof['new_diagnostic_source']:return False
    if config.query_revision not in {'legacy-fixed-v1','continuous-queries-v2','continuous-queries-v3'}:return False
    if config.physics_weights is not None or config.physics_schedule!='constant' or config.gradient_diagnostics_every:return False
    require(all(sha256(root/p)==sha for p,sha in proof['unchanged_files'].items()),'suite compatibility requires unchanged scientific models/losses/numerics/shared optimizer')
    return True


def approved_source(source):
    from types import SimpleNamespace
    return approved_diagnostic_predecessor(SimpleNamespace(diagnostic_source_id=source,query_revision='continuous-queries-v3',physics_weights=None,physics_schedule='constant',gradient_diagnostics_every=0))
