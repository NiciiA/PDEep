"""Interruption control on a small P2 wave space; NOT numerical certification."""
from pdeep.smooth.reference import Reference
from pdeep.smooth.definition import geometry
from pdeep.wave_fidelity.reference import Space
from pdeep.unified.config import DatasetConfig
from pdeep.unified.dataset import quality_check_full
from pdeep.preparation.engine import prepare_dataset,References
from pdeep.preparation.transfer import sync_folder
from pdeep.preparation.portable import read_index,verify_folder

def test_partial_label_transfer_and_exact_manifest_resume(certificate,tmp_path,monkeypatch):
    r=object.__new__(Reference);r.space=Space(geometry('hole',.12),2,speed=1.);r.family='hole';r.h=.12;r.degree=2;r.factors={}
    monkeypatch.setattr(References,'get',lambda self,family:r)
    def control_qc(c,full,norms,config,**kw):
        q=quality_check_full(c,full,norms,config,finer=False);q['scope']='P2 arithmetic interruption control ONLY; not production qualification';return q
    monkeypatch.setattr('pdeep.preparation.engine.quality_check_full',control_qc)
    config=DatasetConfig(preset='smoke',pdes=('wave',),families=('hole',),counts=(4,2,2),stream_updates=3,stream_queries=40000,stream_grid=4,qc_points=8)
    phases=[];quiet=lambda stage,**kw:phases.append((stage,kw))
    source=prepare_dataset(config,tmp_path/'mac',certificate,max_cases=1,max_label_batches=1,progress=quiet,sync_to=tmp_path/'drive')
    assert not source.index['records'] and source.index['partial_cases'] and source.index['status']=='partial'
    assert source.index['persistence']['last_sync_status']=='verified' and source.index['persistence']['dataset_status']=='partial'
    dest=tmp_path/'colab';sync_folder(tmp_path/'drive',dest)
    before={k:v for k,v in source.index['files'].items() if k.startswith('.working/')}
    assert before and not any(dest.glob('.working/*/progress.json')) # Rebuilt from explicit manifest.
    restored=prepare_dataset(config,dest,certificate,manifest=dest/'manifest.json',max_cases=1,progress=quiet,sync_to=tmp_path/'drive')
    assert len(restored.index['records'])==1 and restored.index['status']=='partial'
    assert not restored.index.get('partial_cases')
    assert restored.index['preparation_profile']['encounter_plan_reused']
    verify_folder(tmp_path/'drive')
    remote=read_index(tmp_path/'drive')[1]
    assert remote['records']==restored.index['records'] and not any(k.startswith('.working/') for k in remote['files'])
    assert not any((tmp_path/'drive'/rel).exists() for rel in before)
