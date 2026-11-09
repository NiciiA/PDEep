"""Stage7C bounded persistence tests. Real8 versus4+4 is an explicit CLI check."""
from dataclasses import replace,asdict
from pathlib import Path
import json,subprocess,sys,random
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError,digest
from pdeep.data import calibrate
from pdeep.models import select_architecture
from pdeep.artifact_io import write_artifact,read_manifest,read_payload,atomic_json
from pdeep.artifact_metadata import validate_metadata
from pdeep.artifact_rng import capture_rng,restore_rng
from pdeep.artifact_drive import PersistencePolicy,sync_verified,retention_candidates
from pdeep.inference_artifacts import export_inference,load_inference,verify_inference_examples
from pdeep.diagnostic_artifacts import capture_intermediates,save_diagnostic_bundle
from pdeep.inference_artifacts import inference_examples,unpack_query,unpack_observation
from pdeep.reference import canonical_spec
from pdeep.experiment_suite import SuiteEntry,preview_suite,execute_suite
from pdeep.notebook_state import NotebookLedger

@pytest.fixture(scope='module')
def norm():return calibrate()

@pytest.fixture(scope='module')
def exports(tmp_path_factory,norm):
    root=tmp_path_factory.mktemp('independent-exports');result={}
    for arch in ('A','C','B0','B1','D'):
        model=None if arch=='D' else select_architecture(arch,normalizers=norm)
        result[arch]=export_inference(model,root/arch,architecture=arch,normalizers=norm,run_id='artifact-tests-'+arch)
    return result


def test_manifests_complete_and_noninterchangeable(exports):
    fields={'model_class','architecture_configuration','scientific_contract','source','runtime','save_device','save_precision',
            'problem_specs','equation_vocabulary','geometry_and_boundaries','priors','normalization','data_generation','observation_policy',
            'objective','optimizer_configuration','scheduler_configuration','training_stage','successful_updates','evaluation_protocol',
            'dependencies','files','created_UTC','verification'}
    for arch,path in exports.items():
        record=read_manifest(path,kind='inference_export',architecture=arch)
        assert fields<=record.keys();assert record['verification']['status']=='verified';validate_metadata(record)
        assert {r['name'] for r in record['data_generation']['splits']}=={'train','validation','test'}
        if arch!='D':
            keys=record['component_inventory']['state_schema'];assert any(k.startswith('equation.') for k in keys)
            assert any(k.startswith('observation.') for k in keys)
            if arch=='B1':assert any(k.startswith('spatial.') for k in keys)
            if arch=='B0':assert not any(k.startswith('spatial.') for k in keys)
        with pytest.raises(ValidationError,match='type mismatch'):read_manifest(path,kind='training_checkpoint')
    with pytest.raises(ValidationError,match='architecture'):load_inference(exports['B0'],expected_architecture='B1')

@pytest.mark.parametrize('architecture',['A','C','B0','B1','D'])
def test_independent_inference_process_has_no_training_imports(exports,tmp_path,architecture):
    output=tmp_path/(architecture+'.json')
    command=[sys.executable,'-m','pdeep.persistence_worker','verify-inference','--architecture',architecture,
             '--manifest',str(exports[architecture]),'--output',str(output)]
    r=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
    report=json.loads(output.read_text());assert report['passed'] and report['fresh_process']
    assert report['strict_deterministic_algorithms'] is True
    assert report['training_modules_imported']==[]
    assert max(row['prediction_max_error'] for row in report['cases'])==0
    if architecture=='D':assert max(row['covariance_max_error'] for row in report['cases'])==0


def rewrite(tmp_path,original,*,alter_metadata=None,alter_payload=None):
    record,payload=read_payload(original)
    metadata={k:v for k,v in record.items() if k not in {'schema_version','content_identity','payload_tree_identity','files','verification','artifact_id','created_UTC'}}
    if alter_metadata:alter_metadata(metadata)
    if alter_payload:alter_payload(payload)
    return write_artifact(tmp_path,None,metadata,payload,required=('model','examples'))


def test_strict_missing_weights_checksum_and_payload_fields(exports,tmp_path):
    path=rewrite(tmp_path,exports['B1'],alter_payload=lambda p:p['model'].pop(next(k for k in p['model'] if k.startswith('spatial.'))))
    with pytest.raises(RuntimeError,match='Missing key'):load_inference(path,expected_architecture='B1')
    with pytest.raises(ValidationError,match='required fields'):
        rewrite(tmp_path,exports['A'],alter_payload=lambda p:p.pop('examples'))
    # write_artifact itself refuses structurally unreadable required payloads.

@pytest.mark.parametrize('change,match',[
    (lambda m:m.pop('normalization'),'metadata missing'),
    (lambda m:m['equation_vocabulary'].update(version='incompatible'),'vocabulary'),
    (lambda m:m['architecture_configuration'].update(inverse_track='T2'),'parameterization'),
    (lambda m:m['priors']['heat'][0]['prior']['scale'].__setitem__(0,.9),'prior definitions'),
    (lambda m:m['dependencies']['runtime_solvers'].update(heat='missing.solver'),'solver support'),
    (lambda m:m['component_inventory']['state_schema'].pop(next(iter(m['component_inventory']['state_schema']))),'model components'),
])
def test_incompatible_scientific_manifest_fails_explicitly(exports,tmp_path,change,match):
    path=rewrite(tmp_path,exports['A'],alter_metadata=change)
    with pytest.raises(ValidationError,match=match):load_inference(path,expected_architecture='A')


def test_checksum_failure_before_any_model_construction(exports,tmp_path,monkeypatch):
    path=rewrite(tmp_path,exports['A']);record=read_manifest(path);payload=path.parent/next(iter(record['files']))
    payload.write_bytes(payload.read_bytes()+b'broken')
    monkeypatch.setattr('pdeep.models.select_architecture',lambda *a,**k:pytest.fail('constructor called before checksum verification'))
    with pytest.raises(ValidationError,match='checksum'):load_inference(path,expected_architecture='A')


def test_full_RNG_roundtrip_global_registered_and_monitor_independence():
    random.seed(71);np.random.seed(72);torch.manual_seed(73)
    generators={'numpy':np.random.default_rng(74),'torch':torch.Generator().manual_seed(75),'mt':np.random.Generator(np.random.MT19937(76))}
    state=capture_rng(generators)
    def draws(g):return [random.random(),float(np.random.rand()),float(torch.rand(())),float(g['numpy'].random()),float(torch.rand((),generator=g['torch'])),float(g['mt'].random())]
    expected=draws(generators)
    restored,limitations=restore_rng(state)
    assert draws(restored)==expected
    assert 'python_random' in state and 'numpy_global' in state and 'torch_CPU' in state and 'torch_MPS' in state


def test_atomic_failure_never_replaces_prior_valid_commit(tmp_path,monkeypatch):
    from pdeep import artifact_io
    metadata={'artifact_type':'diagnostic_bundle','architecture_id':'A','run_id':'safe-commit-test'}
    good=write_artifact(tmp_path,'good',metadata,{'tensor':torch.ones(2)},required=('tensor',))
    old=good.read_bytes();old_payload=good.with_suffix('.pt').read_bytes()
    publish=artifact_io._publish
    def fail_manifest(source,dest):
        if Path(dest).suffix=='.json':raise OSError('injected manifest publication failure')
        publish(source,dest)
    monkeypatch.setattr(artifact_io,'_publish',fail_manifest)
    with pytest.raises(OSError):write_artifact(tmp_path,'new',metadata,{'tensor':torch.zeros(2)},required=('tensor',))
    assert good.read_bytes()==old and good.with_suffix('.pt').read_bytes()==old_payload
    assert not (tmp_path/'new.pt').exists() and not (tmp_path/'new.json').exists()
    assert not list(tmp_path.glob('.pending-*')) and not list(tmp_path.glob('*.lock'))
    read_manifest(good)


def test_optional_drive_transaction_and_copy_failure_preserves_local(exports,tmp_path,monkeypatch):
    path=exports['A'];assert sync_verified(path,PersistencePolicy())['Drive_status']=='disabled'
    policy=PersistencePolicy(True,str(tmp_path),True,'filesystem_test')
    success=sync_verified(path,policy);assert success['Drive_verified'] and success['Drive_status']=='completed'
    read_manifest(success['Drive_manifest']);assert path.exists()
    root=tmp_path/'bad';root.mkdir()
    from pdeep import artifact_drive
    copy=artifact_drive.shutil.copyfile
    def bad_copy(src,dst):
        result=copy(src,dst)
        if Path(dst).suffix=='.pt':Path(dst).write_bytes(b'partial')
        return result
    monkeypatch.setattr(artifact_drive.shutil,'copyfile',bad_copy)
    failed=sync_verified(path,replace(policy,drive_mount=str(root)))
    assert failed['Drive_status']=='failed' and not failed['Drive_verified'];read_manifest(path)
    assert not list(root.rglob('.copying-*'))
    assert retention_candidates([Path(str(i)) for i in range(10)],PersistencePolicy())==[]
    assert len(retention_candidates([Path(str(i)) for i in range(10)],PersistencePolicy(retention='suggest_keep_last',keep_last=3)))==7
    with pytest.raises(ValidationError):PersistencePolicy(drive_enabled=True)


def test_diagnostic_attention_memory_and_masks_not_inference(exports,norm,tmp_path):
    example=inference_examples(norm)[-1];spec=canonical_spec('wave');obs=unpack_observation(example['observations']);q=unpack_query(example['queries'])
    for arch in ('A','C','B0','B1','D'):
        wrapper=load_inference(exports[arch],expected_architecture=arch)
        intermediate=capture_intermediates(arch,wrapper.model,spec,obs,q)
        path=save_diagnostic_bundle(tmp_path/arch,architecture=arch,normalizers=norm,run_id='diagnostic-tests-'+arch,model=wrapper.model,inputs=example,intermediates=intermediate)
        record,payload=read_payload(path,kind='diagnostic_bundle');assert not record['safe_to_resume']
        assert torch.equal(payload['inputs']['observations']['valid'],obs.valid)
        if arch=='B1':assert len(payload['intermediates']['spatial_memory'])==3
        if arch=='B0':assert payload['intermediates']['spatial_memory']==()
        if arch in {'A','C'}:assert 'attention' in payload['intermediates']
        with pytest.raises(ValidationError,match='type mismatch'):load_inference(path,expected_architecture=arch)


def test_suite_explicit_actions_preflight_and_no_auto_training(exports,norm,tmp_path):
    entries=tuple(SuiteEntry(a,'inference',str(tmp_path),checkpoint=str(exports[a])) for a in ('A','C','B0','B1','D'))
    view=preview_suite(entries);assert view['execution_order']==['A','C','B0','B1','D']
    assert not any(row['requires_training'] for row in view['entries'])
    models=execute_suite(entries,norm,expected_preview_identity=view['identity']);assert len(models)==5
    with pytest.raises(ValidationError):SuiteEntry('D','fresh',str(tmp_path))
    with pytest.raises(ValidationError):preview_suite((SuiteEntry('B1','inference',str(tmp_path),checkpoint=str(exports['B0'])),))
    from pdeep.train_types import ResolvedTrainConfig
    c=ResolvedTrainConfig('A',source_id='test-suite',run_tag='explicit-fresh')
    training=(SuiteEntry('A','fresh',str(tmp_path),config=c),)
    with pytest.raises(ValidationError,match='training disabled'):execute_suite(training,norm)
    assert not (tmp_path/'A').exists()


def test_notebook_guards_invalidate_downstream_and_config_changes(tmp_path):
    from pdeep.diagnostics import source_identity
    from types import SimpleNamespace
    root=tmp_path;source=source_identity(root)['sha256'];ledger=NotebookLedger(root,source,('one','two','three'));namespace={}
    with pytest.raises(ValidationError,match='previous'): 
        with ledger.cell('two',namespace):pass
    with ledger.cell('one',namespace):namespace['CONFIG']=SimpleNamespace(identity='config1')
    with ledger.cell('two',namespace):namespace['NORMALIZERS']=SimpleNamespace(identity='norm1')
    with ledger.cell('three',namespace):pass
    with ledger.cell('two',namespace):pass
    assert 'three' not in ledger.completed
    namespace['CONFIG']=SimpleNamespace(identity='config2')
    with pytest.raises(ValueError,match='CONFIG'):
        with ledger.cell('three',namespace):pass
    assert not ledger.completed

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3))
        from pdeep.encoding import EquationVocabulary
        self.vocabulary=EquationVocabulary.from_specs(tuple(canonical_spec(p) for p in ('poisson','heat','wave')))

class ToyAdapter:
    def loss(self,model,batch,**kwargs):
        from pdeep.train_types import LossReport
        loss=(model.weight-batch.unknowns.xi[:,0]).square().mean()
        return LossReport(loss,{'toy':loss},{'toy':loss},{'toy':'arithmetic-only injected model'},
                          {'cases':len(batch.specs),'valid_queries':int(batch.queries.valid.sum()),'active_coefficients':int(batch.unknowns.active.sum()),'valid_observations':int(batch.observations.valid.sum())})


def toy_engine(root,norm,**kwargs):
    from pdeep.training import TrainingEngine
    from pdeep.train_types import ResolvedTrainConfig
    c=ResolvedTrainConfig('A',updates=3,microbatch=1,accumulation=1,effective_batch=1,query_count=8,validation_cases_per_pde=1,
                          validation_query_count=8,source_id='toy-artifact-test',run_tag='toy-checkpoint',**kwargs)
    return TrainingEngine(c,norm,root,model=ToyModel(),adapter=ToyAdapter(),probe=False)


def toy_reload(monkeypatch,path,**kwargs):
    from pdeep.train_checkpoint import load_checkpoint
    monkeypatch.setattr('pdeep.models.select_architecture',lambda *a,**k:ToyModel())
    monkeypatch.setattr('pdeep.training.ScientificAdapter',lambda *a:ToyAdapter())
    return load_checkpoint(path,expected_architecture='A',probe=False,**kwargs)


def test_training_checkpoint_complete_histories_selection_and_portable_resume(norm,tmp_path,monkeypatch):
    from pdeep.artifact_tensors import state_identity
    e=toy_engine(tmp_path/'original',norm);e.update()
    e.state.best_validation_metric=.5;e.state.best_validation_step=1;e.state.best_validation_weights=state_identity(e.model)
    first=e.save_checkpoint();meta,payload=read_payload(first,kind='training_checkpoint')
    assert set(('model','optimizer','scheduler','scaler','rng','execution','sampler','histories','validation_manifest'))<=payload.keys()
    assert not payload['AMP'] and payload['scaler'] is None and not payload['state']['accumulation_active']
    assert json.loads((first.parent/'best_validation.json').read_text())['manifest']==first.name
    e.update();second=e.save_checkpoint()
    assert json.loads((first.parent/'last.json').read_text())['manifest']==second.name
    assert json.loads((first.parent/'best_validation.json').read_text())['manifest']==first.name
    with pytest.raises(ValidationError,match='advanced/diverged'):toy_reload(monkeypatch,first)
    restored=toy_reload(monkeypatch,first,run_root=tmp_path/'fork')
    assert restored.state.successful==1 and restored.state.best_validation_metric==.5
    assert restored.sampler.state_dict(restored.state.sampler_cursor)==payload['sampler']
    assert (restored.run_dir/'steps.jsonl').exists() and restored.loaded_artifact_id==meta['artifact_id']
    assert torch.equal(restored.model.weight,payload['model']['weight'])
    restored.update();assert torch.equal(restored.model.weight,e.model.weight)
    e.state.accumulation_active=True
    with pytest.raises(ValidationError,match='boundary'):e.save_checkpoint()


def test_engine_drive_failure_preserves_last_local_and_reports_backlog(norm,tmp_path):
    e=toy_engine(tmp_path/'engine',norm)
    # An explicitly configured, unavailable mount fails only synchronization.
    e.persistence_policy=PersistencePolicy(True,str(tmp_path/'missing'),True,'filesystem_test')
    e.update();path=e.save_checkpoint();read_manifest(path)
    assert e.state.persistence['Drive_status']=='failed' and e.state.persistence['backlog']==[str(path)]
    assert e.state.last_checkpoint==str(path) and e.state.phase=='ready'
    events=[json.loads(line) for line in (e.run_dir/'persistence.jsonl').read_text().splitlines()]
    assert [r['event'] for r in events]==['local_save_started','local_save_completed','Drive_copy_started','Drive_copy_failed']


def test_engine_local_save_failure_does_not_advance_checkpoint_pointer(norm,tmp_path,monkeypatch):
    e=toy_engine(tmp_path,norm);e.update();first=e.save_checkpoint();pointer=(first.parent/'last.json').read_bytes()
    e.update()
    def fail(*a,**k):raise OSError('injected disk/write error')
    monkeypatch.setattr('pdeep.artifact_io._publish',fail)
    with pytest.raises(OSError):e.save_checkpoint()
    assert e.state.last_checkpoint==str(first) and (first.parent/'last.json').read_bytes()==pointer
    read_manifest(first);assert not (first.parent/'update-00000002.json').exists()
    assert e.state.persistence['local_status']=='failed'


def test_remote_cleanup_failure_cannot_interrupt_verified_local_save(norm,tmp_path,monkeypatch):
    e=toy_engine(tmp_path/'engine',norm);mount=tmp_path/'mock_drive';mount.mkdir()
    e.persistence_policy=PersistencePolicy(True,str(mount),True,'filesystem_test')
    def disconnected(*a,**k):raise OSError('simulated disconnected mounted filesystem')
    monkeypatch.setattr('pdeep.artifact_drive.shutil.copyfile',disconnected)
    monkeypatch.setattr('pdeep.artifact_drive.shutil.rmtree',disconnected)
    e.update();path=e.save_checkpoint()
    read_manifest(path)
    assert e.state.last_checkpoint==str(path) and e.state.persistence['Drive_status']=='failed'
    assert e.state.phase=='ready'


def test_failure_bundle_preserves_nonfinite_gradients_and_never_resumes(norm,tmp_path):
    from pdeep.diagnostic_artifacts import save_failure_bundle
    e=toy_engine(tmp_path,norm);batch,_=e.sampler.batch(0)
    e.model.weight.grad=torch.tensor(float('nan'))
    path=save_failure_bundle(e,{'kind':'nonfinite','loss':float('nan')},batch)
    meta,payload=read_payload(path,kind='diagnostic_bundle')
    assert payload['packet']['loss']=='nan' and torch.isnan(payload['gradients']['weight'])
    assert meta['diagnostic_summary']['gradients']['weight']['nonfinite']==1
    from pdeep.train_checkpoint import load_checkpoint
    with pytest.raises(ValidationError,match='type mismatch'):load_checkpoint(path,expected_architecture='A',probe=False)
