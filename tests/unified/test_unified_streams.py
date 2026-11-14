"""Pre-sampling is exactly the counter-keyed continuous law, not bank recycling."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError
from pdeep.smooth.data import Case
from pdeep.smooth.reference import Reference
from pdeep.geometric.domain import mesh_shape,make_shape
from pdeep.wave_fidelity.reference import Space
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.smooth.bank import physical_design
from pdeep.unified.config import DatasetConfig,SCHEMA
from pdeep.unified.dataset import inventory,quality_check_full
from pdeep.unified.streams import encounters,prepare_labels,PlannedReference,save_sampling_space,coordinates_id
from pdeep.unified.data import Provider
from pdeep.artifact_io import write_artifact
from .conftest import EVIDENCE


def test_continuous_plan_prefix_and_no_recycled_draws():
    d=DatasetConfig(preset='smoke',stream_updates=12,stream_queries=16)
    a=encounters(d,inventory(d));b=encounters(replace(d,stream_updates=18),inventory(d))
    assert sum(map(len,a.values()))==48
    for cid,events in a.items():assert events==[e for e in b[cid] if e[0]<12]
    c=Case('square','wave','train',0)
    x=physical_design(c,16,observation=True,set_id=(SCHEMA,'encounter',42,0,0))
    y=physical_design(c,16,observation=True,set_id=(SCHEMA,'encounter',42,1,0))
    assert coordinates_id(*x[:3])!=coordinates_id(*y[:3])

@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_full_trajectory_qc_and_prepared_stream_reload_no_online_solver(certificate,tmp_path,monkeypatch,pde):
    # Small P2 numerical control, NOT a production qualification or research dataset.
    from pdeep.smooth.certification import normalizers
    from pdeep.unified.provenance import reference_execution_identity
    r=object.__new__(Reference);r.space=Space(mesh_shape(make_shape('square'),.35),2,speed=1.);r.family='square';r.h=.35;r.degree=2;r.factors={}
    c=Case('square',pde,'train',0);full=FixedReference(c,reference_object=r);norms=normalizers(certificate)
    d=DatasetConfig(preset='smoke',stream_updates=3,stream_queries=16,stream_validation_queries=8,stream_grid=4,qc_points=8)
    qc=quality_check_full(c,full,norms,d,finer=False);assert qc['passed'],qc
    labels=prepare_labels(c,full,d,[(0,0,64),(1,0,64)])
    space=save_sampling_space(full,tmp_path,reference_execution_identity())
    payload={'case':c.manifest,'qc':qc,'labels':labels,'sampling_space':str(space.relative_to(tmp_path))}
    path=write_artifact(tmp_path/'cases','small-control',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'small-P2-control'},payload)
    ds=SimpleNamespace(root=tmp_path,config=d,certificate=certificate,norms=norms,index={'records':{c.case_id:str(path.relative_to(tmp_path))}},require_case=lambda c:None)
    ds.cache=PlannedReference(ds);provider=Provider(ds)
    def fail(*a,**kw):pytest.fail('online reference evaluation invoked a solver')
    monkeypatch.setattr(FixedReference,'__init__',fail)
    # Actual counter-keyed provider's N may differ from explicit event; use fixed64
    # to check both encounters and exact lookup. The plan stores their original draws.
    a,ai=provider.batch((c,),cursor=0,count=16,observation_count=64)
    b,bi=provider.batch((c,),cursor=1,count=16,observation_count=64)
    assert not torch.equal(a.observations.position,b.observations.position)
    design=physical_design(c,16,set_id=(SCHEMA,'supervised',42,0,0))
    expected=full.evaluate(*design[:3]);np.testing.assert_allclose(a.queries.value[0,:,0],expected,rtol=1e-6,atol=1e-8)
    error=float(np.max(abs(ds.cache.evaluate(c,*design[:3])-expected)));assert error==0.
    with pytest.raises(ValidationError,match='not prepared'):ds.cache.evaluate(c,design[0],np.full(16,.111),design[2])
    EVIDENCE[pde+'_planned_draw_label_max_error']=error
    EVIDENCE[pde+'_full_trajectory_QC']=qc['solver']
