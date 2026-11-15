"""Reviewed medium transition and full-shape capacity boundaries."""
from dataclasses import replace
from pathlib import Path
import json
import torch
from pdeep.unified_modes.model import create,ConditionedModel
from pdeep.geometric.models import ModelConfig
from pdeep.unified_modes.compatibility import approved_extension,resume_configurations
from pdeep.unified_modes.training import Engine,resume
from pdeep.unified_modes.dashboard import Dashboard
from pdeep.unified_modes.config import extension_identity
from pdeep.unified.config import implementation_identity
from pdeep.unified_bayes.compatibility import approved_source
from pdeep.artifact_io import read_payload,write_artifact
from pdeep.artifact_tensors import state_identity
from .conftest import config

def test_medium_initialization_bitwise_and_reviewed_sources(dataset):
    for arch in ('A_GEO','C_GEO'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);old=ConditionedModel(ModelConfig.preset(arch,'medium'),dataset.norms)
        new=create(arch,dataset.norms,'medium',seed=42)
        assert state_identity(old)==state_identity(new)
    proof=json.loads(Path('src/pdeep/unified_modes/large_transition.json').read_text())
    assert proof['after']==extension_identity() and proof['base_after']==implementation_identity()
    assert approved_extension(proof['before'],'medium') and approved_extension(proof['before'],'verification')
    assert not approved_extension(proof['before'],'large') and not approved_extension('unreviewed','medium')
    assert approved_source('16b77c6df38338f85ccd178fbf103969812021f0b6415515926e5fec3d661373')

def test_reviewed_legacy_resume_keeps_config_optimizer_and_order(dataset,tmp_path):
    """Synthetic legacy-source fixture, exact original widths; never relabel real artifacts."""
    proof=json.loads(Path('src/pdeep/unified_modes/large_transition.json').read_text())
    old_source='16b77c6df38338f85ccd178fbf103969812021f0b6415515926e5fec3d661373'
    cfg=config(dataset,extension_source=proof['before'],source_id=old_source)
    e=Engine(cfg,dataset,tmp_path/'old',monitor=Dashboard(notebook=False));e.update();path=e.save_checkpoint()
    record,payload=read_payload(path);record['implementation_identity']=old_source;record.pop('parameter_counts')
    for key in ('artifact_id','schema_version','content_identity','payload_tree_identity','created_UTC','files','verification'):record.pop(key,None)
    fixture=write_artifact(tmp_path/'legacy-fixture','reviewed-previous-source',record,payload,required=tuple(payload))
    preview=replace(cfg,extension_source=extension_identity(),source_id=implementation_identity())
    selected=resume_configurations((preview,),{cfg.model_key:fixture})[0]
    assert selected.identity==cfg.identity
    restored=resume(fixture,selected,dataset,tmp_path/'restored',monitor=Dashboard(notebook=False))
    assert restored.scheduler.state_dict()==e.scheduler.state_dict()
    assert restored.sampler.state_dict(1)==e.sampler.state_dict(1)
    e.update();restored.update();assert state_identity(e.model)==state_identity(restored.model)
