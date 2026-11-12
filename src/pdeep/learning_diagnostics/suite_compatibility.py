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
    path=root/'certifications/G1_smooth_v1/portable/diagnostic_suite_execution.json'
    if not path.exists():return False
    proof=json.loads(path.read_text())
    require(digest({k:v for k,v in proof.items() if k!='identity'})==proof['identity'],'suite compatibility proof checksum mismatch')
    from .engine import diagnostic_source
    if config.diagnostic_source_id not in proof['approved_predecessors'] or diagnostic_source()!=proof['new_diagnostic_source']:return False
    if config.query_revision not in {'legacy-fixed-v1','continuous-queries-v2'}:return False
    require(all(sha256(root/p)==sha for p,sha in proof['unchanged_files'].items()),'suite compatibility requires unchanged scientific models/losses/numerics/shared optimizer')
    return True
