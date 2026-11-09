"""Routine workflow guards/integration; no research training in unit tests."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from pdeep.core import ValidationError
from pdeep.notebook_workflows import (initialize,prepare_training,construct_models,launch_training,plan_table,
    choose_device,load_selected,load_evaluation_selection,make_evaluation_plan,preserved_verification,assert_plan_settings)
from pdeep.inference_artifacts import export_inference
from pdeep.models import select_architecture

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def context():return initialize(ROOT,device='cpu')

@pytest.fixture(scope='module')
def untrained_export(context,tmp_path_factory):
    directory=tmp_path_factory.mktemp('workflow-untrained')
    model=select_architecture('B0',normalizers=context.normalizers)
    path=export_inference(model,directory,architecture='B0',normalizers=context.normalizers,run_id='workflow-test-untrained')
    return path


def test_preview_research_no_engine_or_directory_and_original_defaults(context,monkeypatch):
    p=prepare_training(context,models=('B1',),run_name='preview-test-never-start')
    def forbidden(*a,**k):pytest.fail('preview constructed a training engine')
    monkeypatch.setattr('pdeep.training.TrainingEngine',forbidden)
    assert launch_training(p,run_mode='preview')=={}
    assert not Path(p.preview['entries'][0]['outputs']['run_directory']).exists()
    c=p.entries[0].config
    assert c.updates==10000 and c.learning_rate==3e-4 and c.microbatch==8 and c.accumulation==2 and c.effective_batch==16
    assert c.validation_cases_per_pde==128 and c.query_count==128 and c.scheduler=='warmup_then_cosine'
    assert 'B1' in plan_table(p) and '10000' in plan_table(p)


@pytest.mark.parametrize('architecture',['A','B0','B1','C'])
def test_construct_only_selected_architecture(context,architecture):
    p=prepare_training(context,models=(architecture,),preset='smoke',updates=2)
    models=construct_models(p)
    assert tuple(models)==(architecture,)
    assert sum(v.numel() for v in models[architecture].parameters())==p.preview['entries'][0]['estimated_parameters']
    assert p.entries[0].config.microbatch==(1 if architecture=='C' else 8)
    assert p.entries[0].config.effective_batch==16


def test_config_guard_and_single_action(context):
    p=prepare_training(context,models=('B0',),preset='smoke',updates=2)
    from pdeep.artifact_drive import PersistencePolicy
    settings=dict(p.settings);settings['persistence']=PersistencePolicy()
    assert_plan_settings(p,**settings)
    settings['updates']=3
    with pytest.raises(ValidationError,match='settings changed'):assert_plan_settings(p,**settings)
    with pytest.raises(ValidationError,match='RUN_MODE'):launch_training(p,run_mode='implicit')
    with pytest.raises(ValidationError,match='resume requires'):launch_training(p,run_mode='resume')
    with pytest.raises(ValidationError,match='evaluation-only'):prepare_training(context,models=('D',))
    with pytest.raises(ValidationError):prepare_training(context,models=('B0','B0'))


def test_device_policy_explicit_and_no_fallback():
    chosen,message=choose_device('cpu');assert chosen=='cpu' and 'explicit' in message
    assert choose_device('auto',saved_device='cpu')[0]=='cpu'
    with pytest.raises(ValidationError,match='preserves'):choose_device('cpu',saved_device='mps')


def test_no_smoke_or_untrained_substitution(context,untrained_export):
    with pytest.raises(ValidationError,match='not a completed research'):load_selected(untrained_export,'B0')
    model,norm,meta,status,path=load_selected(untrained_export,'B0',allow_partial=True)
    assert status=='untrained' and meta['successful_updates']==0 and norm.identity==context.normalizers.identity
    with pytest.raises(ValidationError,match='architecture'):load_selected(untrained_export,'B1',allow_partial=True)
    with pytest.raises(ValidationError,match='not found'):load_evaluation_selection(context,{'B1':'/no/such/model.json'},allow_partial=True)
    methods=load_evaluation_selection(context,{})
    assert [m.name for m in methods]==['D'] and methods[0].status=='exact; no training'
    methods=load_evaluation_selection(context,{'B0':str(untrained_export)},allow_partial=True)
    assert [m.name for m in methods]==['B0','D'] and methods[0].training['successful_updates']==0


def test_plan_common_grid_and_loaded_settings(context,tmp_path):
    from pdeep.evaluation_plan import EvaluationPlan
    p,c=make_evaluation_plan(context,pdes=('poisson','wave'),case_ids=(0,))
    path=p.save(tmp_path);same,_=make_evaluation_plan(context,pdes=('poisson','wave'),plan_path=path)
    assert p==same and len(p.spatial_points)==4096 and p.dynamic_times[-1]==1
    assert not p.query('poisson').time_present.any() and p.query('wave',1).valid.sum()==9*4096
    with pytest.raises(ValidationError,match='differs'):make_evaluation_plan(context,pdes=('heat',),plan_path=path)


def test_old_verification_revalidated_without_retraining(context,monkeypatch):
    r=preserved_verification(ROOT)
    assert r['passed'] and r['UI_revalidation_only'] and r['proof_creation_source_identity']==r['source_identity']
    assert r['verified_against_source_identity']==context.source_id
    from pdeep import notebook_workflows
    original=notebook_workflows.source_identity
    def changed(root):
        source=original(root);source['files']['src/pdeep/reference.py']='modified'
        return source
    monkeypatch.setattr(notebook_workflows,'source_identity',changed)
    with pytest.raises(ValidationError,match='implementation changed'):preserved_verification(ROOT)


def test_portable_resume_is_verified_and_staged_without_overwrite(context,tmp_path):
    from pdeep.notebook_workflows import stage_resume_checkpoint
    from pdeep.artifact_io import read_manifest
    import shutil
    pointer=json.loads((ROOT/'reports/latest_persistence.json').read_text());record=json.loads((ROOT/pointer['report']).read_text())
    source=Path(record['neural_checks']['B0']['end_to_end']['checkpoint']);meta=read_manifest(source)
    portable=tmp_path/'portable';portable.mkdir()
    shutil.copyfile(source,portable/source.name)
    for name in meta['files']:shutil.copyfile(source.parent/name,portable/name)
    local=replace(context,root=tmp_path/'local')
    staged=stage_resume_checkpoint(local,portable/source.name,'B0')
    assert staged.parent.name=='checkpoints' and staged.parent.parent.name==meta['run_id']
    assert read_manifest(staged)['artifact_id']==meta['artifact_id']
    before=staged.stat().st_mtime_ns
    assert stage_resume_checkpoint(local,portable/source.name,'B0')==staged
    assert staged.stat().st_mtime_ns==before


def test_launch_edit_can_autosave_but_implementation_edits_invalidate(context,monkeypatch):
    from pdeep import notebook_workflows
    original=notebook_workflows.source_identity
    def notebook_edit(root):
        source=original(root);source['sha256']='notebook-autosaved-start-action'
        source['files']['notebooks/02_train.ipynb']='RUN_MODE=train'
        return source
    monkeypatch.setattr(notebook_workflows,'source_identity',notebook_edit)
    context.require_current()
    def implementation_edit(root):
        source=notebook_edit(root);source['files']['src/pdeep/reference.py']='changed implementation'
        return source
    monkeypatch.setattr(notebook_workflows,'source_identity',implementation_edit)
    with pytest.raises(ValidationError,match='implementation changed'):context.require_current()
