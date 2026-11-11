"""Audited workflow-only transition: no model/loss/numerical change is allowed.

The existing repaired implementation and its already-approved predecessor stay
readable. Exact resume still requires checkpoint runtime, device, config, data,
RNG and certificate compatibility in the original loader.
"""
from pathlib import Path
import json
from ..core import digest,require
from ..artifact_io import sha256

ROOT=Path(__file__).resolve().parents[3]
PROOF=ROOT/'certifications/G1_smooth_v1/portable/diagnostic_workflow_execution.json'


def approved_workflow_predecessor(identity):
    from ..smooth.training import implementation_identity
    paths=[PROOF.parent/'continuous_query_execution.json',PROOF]
    for path in paths:
        if not path.exists():continue
        r=json.loads(path.read_text())
        require(digest({k:v for k,v in r.items() if k!='identity'})==r['identity'],'workflow compatibility evidence checksum mismatch')
        if implementation_identity()!=r['new_implementation'] or identity not in r['approved_predecessors']:continue
        require(all(sha256(ROOT/p)==v for p,v in r['unchanged_files'].items()),'execution compatibility requires unchanged models/losses/numerics/shared optimizer')
        return True
    return False
