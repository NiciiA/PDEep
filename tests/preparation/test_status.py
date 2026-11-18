import pytest
from pdeep.preparation.status import status

def test_exact_manifest_audit_is_read_only_and_reports_all_tasks(prepared,monkeypatch):
    from pdeep.smooth.reference import Reference
    def forbidden(*a,**k):pytest.fail('status constructed a numerical reference')
    monkeypatch.setattr(Reference,'__init__',forbidden)
    before=prepared.path.read_bytes()
    brief=status(prepared.path)
    assert brief['recorded_completed']==8 and brief['independently_verified_completed'] is None
    audit=status(prepared.path,verify='payloads',mirror=prepared.path)
    assert audit['independently_verified_completed']==8 and audit['mirror_indexed_cases']==8
    assert not audit['verification_failures'] and audit['per_task'][0]['state']=='complete'
    assert prepared.path.read_bytes()==before
    with pytest.raises(ValueError):status(prepared.path,verify='unknown')
