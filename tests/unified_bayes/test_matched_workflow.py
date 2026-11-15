from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest,torch
from pdeep.artifact_drive import PersistencePolicy
from pdeep.artifact_io import write_artifact,atomic_json,read_manifest
from pdeep.unified.config import DatasetConfig,MODELS
from pdeep.unified.training import Engine
from pdeep.unified_bayes.operators import ForwardOperators,Coordinates
from pdeep.unified_bayes.evaluation import evaluate_case,observation_design,query_design,heldout_design,evaluate_classical,load_classical,SCHEMA
from pdeep.unified_bayes.neural import evaluate_neural
from pdeep.unified_bayes.comparison import summarize
from pdeep.unified_bayes.plots import figures
from pdeep.unified_bayes.storage import ArtifactStore,restore_index
from pdeep.unified_bayes.dashboard import Dashboard
from tests.unified.conftest import ArithmeticDataset,config
from .test_operators import control
from .conftest import EVIDENCE


def test_matched_six_methods_no_optimizer_and_seven_figures(certificate,tmp_path,monkeypatch):
    # Small arithmetic target fixture verifies plumbing, not reconstruction accuracy.
    d=ArithmeticDataset(certificate,DatasetConfig(preset='smoke',pdes=('poisson',),families=('hole',)))
    c=d.cases('test')[0];op=ForwardOperators(tmp_path/'op',certificate,_reference_factory=control,_control_identity='NONPRODUCTION-plumbing-P2')
    designs=[observation_design(c,8,d.config.seed),query_design(c,4),heldout_design(c,d.config.seed)]
    op.prepare(c.pde,c.family,{str(i):Coordinates.design(v) for i,v in enumerate(designs)})
    classical=evaluate_case(c,d,op,8,size=4,save_example=True);assert not classical['truth_used_for_conditioning']
    with pytest.raises(ValueError,match='refuses'):evaluate_classical(d,op,tmp_path/'must-not-evaluate')
    configs=tuple(config(d,k) for k in MODELS);paths={}
    for cfg in configs:
        e=Engine(cfg,d,tmp_path/'checkpoints',monitor=Dashboard(notebook=False));paths[cfg.model_key]=str(e.save_checkpoint())
    def forbidden(*a,**k):pytest.fail('evaluation constructed optimizer')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    neural=evaluate_neural(paths,configs,d,tmp_path/'neural',selection='latest',counts=(8,),size=4,max_cases_per_task=1,diagnostics=False,monitor=Dashboard(notebook=False))
    rows=classical['rows']+neural['rows'];report=summarize(rows,replicates=50);assert len({r['method'] for r in rows})==6
    assert len({(r['case_id'],r['observation_identity'],r['query_identity']) for r in rows})==1 and neural['optimizer_updates']==0
    # Save/reload verified classical payload through the production loader.
    path=write_artifact(tmp_path/'results','case',{'artifact_type':'diagnostic_bundle','architecture_id':'G1_BAYES_RESULTS','run_id':'bounded','evaluation_identity':'bounded','case_id':c.case_id,'sensor_count':8},classical)
    index=tmp_path/'index.json';atomic_json(index,{'schema':SCHEMA,'evaluation_identity':'bounded','case_results':[str(path)]});loaded=load_classical(index)
    plots=figures(report,loaded,neural,tmp_path/'plots');assert len(plots)==7 and all(Path(p).stat().st_size>1000 for p in plots)
    assert len(loaded['coverage_by_geometry'])==1
    from pdeep.unified_bayes.workflow import combined
    output=combined(index,neural,tmp_path/'combined')
    assert len(output['report']['tables'])>6 and output['report']['cost'][2]['amortized_compute_seconds_including_shared_preparation']>0
    # Idempotent replay reads verified per-case results; no second neural inference.
    import pdeep.unified_bayes.neural as module
    monkeypatch.setattr(module,'predict',forbidden)
    again=evaluate_neural(paths,configs,d,tmp_path/'neural',selection='latest',counts=(8,),size=4,max_cases_per_task=1,diagnostics=False,monitor=Dashboard(notebook=False))
    assert len(again['rows'])==3
    EVIDENCE['six_method_plumbing']={'split':'test','PDE':'poisson','geometry':'hole','cases':1,'sensors':8,'queries':len(designs[1][0]),'figures':7,'optimizer_updates':0,'scope':'untrained neural/zero arithmetic targets; NOT G1 test accuracy'}


def test_baseline_store_known_restore_no_drive_search(tmp_path,monkeypatch):
    root=tmp_path/'local';drive=tmp_path/'drive';drive.mkdir();policy=PersistencePolicy(True,str(drive),True,backend='filesystem_test',drive_root=str(drive))
    store=ArtifactStore(root,policy);path=write_artifact(root/'classical/cases','one',{'artifact_type':'diagnostic_bundle','architecture_id':'G1_BAYES_RESULTS','run_id':'bounded'},{'tensor':torch.ones(2)});status=store(path)
    index=root/'classical/index.json';atomic_json(index,{'case_results':[str(path)]});store.index(index);remote=store.record['indices']['classical/index.json']['Drive_path'];path.unlink();path.with_suffix('.pt').unlink()
    def forbidden(*a,**k):pytest.fail('recursive/enumerated Drive lookup')
    monkeypatch.setattr(Path,'rglob',forbidden);monkeypatch.setattr(Path,'glob',forbidden)
    restored=restore_index(remote,Path(store.policy.drive_root)/'persistence.json',tmp_path/'restored');saved=json.loads(restored.read_text())
    assert read_manifest(saved['case_results'][0])['verification']['status']=='verified' and status['backend']=='filesystem_test'
    EVIDENCE['baseline_persistence']='verified filesystem_test local-first copy and exact-index restore; actual Google Drive NOT tested'
