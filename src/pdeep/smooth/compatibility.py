"""Narrow, explicit execution compatibility for the verified input-loader repair.

Original artifacts/config/data identities are not rewritten. Only the exact
recorded predecessor implementation can use the new loader, after current-runtime
numerical qualification, same dependency versions and unchanged model/loss files.
"""
from pathlib import Path
import json
from ..core import digest,require
from .training import implementation_identity

PROOF=Path(__file__).resolve().parents[3]/'certifications/G1_smooth_v1/portable/execution.json'

def approved_predecessor(identity):
    if identity==implementation_identity():return True
    from ..learning_diagnostics.execution_compatibility import approved_workflow_predecessor
    if approved_workflow_predecessor(identity):return True
    if not PROOF.exists():return False
    r=json.loads(PROOF.read_text())
    require(r['identity']==digest({k:v for k,v in r.items() if k!='identity'}),'execution compatibility proof checksum mismatch')
    return identity==r['predecessor_implementation'] and implementation_identity()==r['repaired_implementation']

def execution_certificate(certificate):
    from .certification import authorize
    try:authorize(certificate);return certificate
    except ValueError:
        from ..learning_diagnostics.certificates import resolve_certification
        current=resolve_certification(requested_capabilities={'certification_id':None}).require()
        require(current.get('parent_certification_id')==certificate['certification_id'],'checkpoint certificate is not the verified predecessor; no implicit scientific migration')
        require(current['reference_policy']==certificate['reference_policy'] and current['normalizers']['identity']==certificate['normalizers']['identity'],'checkpoint physics/reference/normalization differs')
        # Same numerical versions are required for exact continuation. A changed
        # environment can use portable evaluation, not silently claim exact resume.
        require(current['source_revision']['dependencies']==certificate['source_revision']['dependencies'],'exact predecessor resume requires its original numerical dependency versions; portable evaluation is separate')
        return current
