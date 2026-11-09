"""Bounded local tests. Simulated Colab/Drive is explicitly NOT hosted verification."""
import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace, ModuleType

import pytest
from pdeep import colab_setup as setup
from pdeep.artifact_drive import PersistencePolicy,sync_verified
from pdeep.artifact_io import read_manifest
from pdeep.notebook_workflows import initialize,prepare_training,stage_resume_checkpoint

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/example/pdeep-test.git'


def git(*args,cwd=None,env=None):
    return subprocess.run(['git',*args],cwd=cwd,env=env,check=True,capture_output=True,text=True).stdout.strip()


@pytest.fixture
def remote(tmp_path):
    # Real Git transport/refs, isolated local URL rewrite; no network or credentials.
    source=tmp_path/'source';source.mkdir()
    git('init','-b','main',str(source))
    for name in ('pyproject.toml','src/pdeep/__init__.py','src/pdeep/colab_setup.py','docs/scientific_contract_v1.md'):
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
    git('add','.',cwd=source)
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','initial',cwd=source)
    sha=git('rev-parse','HEAD',cwd=source)
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','tag','-a','v1','-m','release',cwd=source)
    bare=tmp_path/'remote.git';git('clone','--bare',str(source),str(bare))
    config=tmp_path/'gitconfig'
    config.write_text('[url "'+bare.as_uri()+'"]\n\tinsteadOf = '+URL+'\n[protocol "file"]\n\tallow = always\n')
    env={'GIT_CONFIG_GLOBAL':str(config),'GIT_CONFIG_NOSYSTEM':'1'}
    return SimpleNamespace(source=source,bare=bare,sha=sha,env=env,destination=tmp_path/'checkout')


def test_clone_pinned_and_existing_tag_and_branch(remote):
    r=remote
    report=setup.ensure_checkout(URL,r.sha,r.destination,git_env=r.env)
    assert report['git_commit']==r.sha and not report['existing_checkout']
    assert report['credentials_recorded'] is False
    for ref in (r.sha,'v1','main','refs/tags/v1'):
        same=setup.ensure_checkout(URL,ref,r.destination,git_env=r.env)
        assert same['git_commit']==r.sha and same['existing_checkout'] and not same['worktree_modified_by_setup']
    assert git('status','--porcelain',cwd=r.destination)==''


def test_existing_dirty_wrong_origin_and_revision_never_reset(remote):
    r=remote;setup.ensure_checkout(URL,r.sha,r.destination,git_env=r.env)
    f=r.destination/'notes.txt';f.write_text('preserve me')
    with pytest.raises(ValueError,match='modifications'):setup.ensure_checkout(URL,r.sha,r.destination,git_env=r.env)
    assert f.read_text()=='preserve me';f.unlink()
    with pytest.raises(ValueError,match='origin differs'):setup.ensure_checkout('https://github.com/other/project',r.sha,r.destination,git_env=r.env)
    with pytest.raises(ValueError,match='nothing was reset'):setup.ensure_checkout(URL,'0'*40,r.destination,git_env=r.env)
    assert git('rev-parse','HEAD',cwd=r.destination)==r.sha
    (r.source/'new.txt').write_text('advanced branch')
    git('add','.',cwd=r.source);git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','next',cwd=r.source)
    git('push',str(r.bare),'main',cwd=r.source)
    with pytest.raises(ValueError,match='nothing was reset'):setup.ensure_checkout(URL,'main',r.destination,git_env=r.env)
    assert git('status','--porcelain',cwd=r.destination)==''


def test_missing_structure_and_existing_directory(remote,tmp_path):
    r=remote;git('rm','docs/scientific_contract_v1.md',cwd=r.source)
    git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','bad',cwd=r.source)
    git('push',str(r.bare),'main',cwd=r.source)
    with pytest.raises(ValueError,match='structure'):setup.ensure_checkout(URL,'main',r.destination,git_env=r.env)
    assert not r.destination.exists()
    r.destination.mkdir();(r.destination/'keep').write_text('keep')
    with pytest.raises(ValueError,match='not a Git checkout'):setup.ensure_checkout(URL,'main',r.destination,git_env=r.env)
    assert (r.destination/'keep').read_text()=='keep'


@pytest.mark.parametrize('url',['SET_YOUR_GITHUB_REPOSITORY_URL','https://SECRET@github.com/x/y','https://x:SECRET@github.com/x/y','https://github.com/x/y?token=SECRET','file:///tmp/a','https://github.com/x/y\nSECRET'])
def test_reject_credentials_and_placeholder(url,tmp_path):
    with pytest.raises(ValueError) as error:setup.ensure_checkout(url,'main',tmp_path/'checkout')
    assert 'SECRET' not in str(error.value)


def test_auth_failure_and_unsupported_python_suppress_raw_output(tmp_path,monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='Authentication failed TOKEN_SECRET',stdout=''))
    with pytest.raises(RuntimeError,match='authentication failed') as error:setup.ensure_checkout(URL,'main',tmp_path/'new')
    assert 'TOKEN_SECRET' not in str(error.value) and not (tmp_path/'new').exists()
    monkeypatch.setattr(sys,'version_info',(3,9))
    with pytest.raises(RuntimeError,match='Python >=3.10'):setup.ensure_checkout(URL,'main',tmp_path/'new')


def test_detection_local_no_google_import_and_local_storage_ignores_colab_settings(monkeypatch,tmp_path):
    monkeypatch.delitem(sys.modules,'google.colab',raising=False)
    assert not setup.running_in_colab(SimpleNamespace())
    shell=type('Shell',(),{'__module__':'google.colab._shell'})()
    assert setup.running_in_colab(shell)
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kw):
        if name.startswith('google'):pytest.fail('local path imported Google')
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    monkeypatch.setattr(setup,'ensure_checkout',lambda *a,**kw:pytest.fail('local clone'))
    monkeypatch.setattr(setup,'install_checkout',lambda *a,**kw:pytest.fail('local install'))
    result=setup.finish_setup(is_colab=False,use_drive=True,drive_root='/never',local_output_dir='/never')
    assert not result['persistence'].drive_enabled and result['local_output_dir']==str(ROOT/'artifacts')
    assert setup.verify_import(ROOT)==str(ROOT/'src/pdeep')
    with pytest.raises(RuntimeError,match='another source'):setup.verify_import(tmp_path)


def test_editable_install_command_and_failures(monkeypatch):
    calls=[]
    def run(args,**kw):calls.append((args,kw));return SimpleNamespace(returncode=0)
    assert setup.install_checkout(ROOT,runner=run)==str(ROOT/'src/pdeep')
    assert calls[0][0]==[sys.executable,'-m','pip','install','-e',str(ROOT)]
    assert calls[0][1]['capture_output']
    with pytest.raises(RuntimeError,match='output is not logged') as error:
        setup.install_checkout(ROOT,runner=lambda *a,**kw:SimpleNamespace(returncode=1,stderr='SECRET'))
    assert 'SECRET' not in str(error.value)
    monkeypatch.setattr(sys,'version_info',(3,9))
    with pytest.raises(RuntimeError,match='Unsupported Python'):setup.install_checkout(ROOT,runner=run)


def test_real_editable_install_isolated_environment(tmp_path):
    # Actual offline pip installation/import in its own environment, NOT Colab.
    env=tmp_path/'venv'
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True,capture_output=True,timeout=60)
    python=env/'bin/python'
    # --system-site-packages inherits the base interpreter, not the test venv.
    # Make its installed build backend/dependencies available without downloads.
    import site
    child_env={**os.environ,'PYTHONPATH':os.pathsep.join(site.getsitepackages()),'PIP_NO_INDEX':'1'}
    subprocess.run([str(python),'-m','pip','install','--no-deps','--no-build-isolation','-e',str(ROOT)],env=child_env,check=True,capture_output=True,timeout=60)
    result=subprocess.run([str(python),'-c','import pdeep; from pathlib import Path; print(Path(pdeep.__file__).resolve())'],env=child_env,check=True,capture_output=True,text=True,timeout=30)
    assert result.stdout.strip()==str(ROOT/'src/pdeep/__init__.py')


def test_cuda_absence_is_explicit_no_fallback(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='no CPU fallback'):setup.validate_cuda()


@pytest.fixture
def simulated_drive(tmp_path,monkeypatch):
    # API/mount simulation only: never claim hosted Google Drive success.
    mount=tmp_path/'drive';mount.mkdir();calls=[]
    google=ModuleType('google');colab=ModuleType('google.colab')
    colab.drive=SimpleNamespace(mount=lambda p:calls.append(p));google.colab=colab
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.colab',colab)
    monkeypatch.setattr(os.path,'ismount',lambda p:Path(p)==mount)
    return mount,calls


def test_simulated_mount_storage_and_missing_mount(simulated_drive,tmp_path,monkeypatch):
    mount,calls=simulated_drive;root=mount/'MyDrive/PDEInverseResearch'
    args=dict(is_colab=True,use_drive=True,drive_root=root,local_output_dir=tmp_path/'local',project_dir=ROOT,mount_point=mount)
    result=setup.configure_storage(**args)
    assert calls==[str(mount)] and result['Drive_destination_ready']
    assert result['persistence'].drive_root==str(root)
    assert all((root/name).is_dir() for name in ('checkpoints','inference','reports'))
    assert (tmp_path/'local/training_runs').is_dir()
    assert 'verification occurs only after' in result['message']
    monkeypatch.setattr(os.path,'ismount',lambda p:False)
    with pytest.raises(RuntimeError,match='without a mounted'):setup.configure_storage(**args)
    with pytest.raises(ValueError,match='outside the Drive'):setup.configure_storage(**{**args,'local_output_dir':root})
    with pytest.raises(ValueError,match='inside the configured'):setup.configure_storage(**{**args,'drive_root':tmp_path/'elsewhere'})


def existing_checkpoint(architecture='B0'):
    p=json.loads((ROOT/'reports/latest_persistence.json').read_text());r=json.loads((ROOT/p['report']).read_text())
    return Path(r['neural_checks'][architecture]['end_to_end']['checkpoint'])


def test_verified_filesystem_copy_custom_root_failure_and_portable_resume(tmp_path,monkeypatch):
    path=existing_checkpoint();meta=read_manifest(path);mount=tmp_path/'simulated-drive';mount.mkdir();root=mount/'MyDrive/PDEInverseResearch'
    policy=PersistencePolicy(True,str(mount),True,'filesystem_test',drive_root=str(root))
    copied=sync_verified(path,policy)
    assert copied['Drive_verified'] and copied['backend']=='filesystem_test'
    remote=Path(copied['Drive_manifest'])
    assert remote.parent.parent==root/'checkpoints'/meta['run_id']
    context=initialize(ROOT,device='cpu',output_dir=tmp_path/'local')
    staged=stage_resume_checkpoint(context,remote,'B0')
    assert staged.is_relative_to(tmp_path/'local/training_runs') and read_manifest(staged)['artifact_id']==meta['artifact_id']
    plan=prepare_training(context,models=('B0',),device='cpu',action='resume',resume_checkpoint=remote,persistence=policy)
    assert plan.entries[0].config.identity==meta['config_identity']
    from pdeep.core import ValidationError
    with pytest.raises(ValidationError,match='architecture'):prepare_training(context,models=('B1',),device='cpu',action='resume',resume_checkpoint=remote)
    with pytest.raises(ValidationError,match='PDE families'):prepare_training(context,models=('B0',),pde_families=('heat',),device='cpu',action='resume',resume_checkpoint=remote)
    # A new destination's remote failure cannot remove the source local artifact.
    failure=replace(policy,drive_root=str(mount/'another'))
    monkeypatch.setattr(shutil,'copyfile',lambda *a,**kw:(_ for _ in ()).throw(OSError('simulated disconnected Drive')))
    status=sync_verified(path,failure)
    assert status['Drive_status']=='failed' and not status['Drive_verified']
    assert read_manifest(path)['artifact_id']==meta['artifact_id']


def test_prominent_sync_warning_is_separate_from_local_save():
    from pdeep.workflow_plotting import persistence_warning
    state=SimpleNamespace(last_checkpoint='/local/checkpoint',persistence={'Drive_status':'failed','local_status':'completed','error':'disconnect','backlog':['/local/checkpoint']})
    warning=persistence_warning(state)
    assert 'DRIVE SYNC FAILED' in warning and 'temporary' in warning and '/local/checkpoint' in warning
    state.persistence['Drive_status']='completed';assert persistence_warning(state) is None


def test_notebook_bootstrap_exact_helper_and_safe_configuration():
    n=json.loads((ROOT/'notebooks/02_train.ipynb').read_text())
    code=''.join(n['cells'][2]['source']);actual=next(a for a in ast.parse(code).body if isinstance(a,ast.FunctionDef) and a.name=='ensure_checkout')
    expected=next(a for a in ast.parse((ROOT/'src/pdeep/colab_setup.py').read_text()).body if isinstance(a,ast.FunctionDef) and a.name=='ensure_checkout')
    assert ast.dump(actual,include_attributes=False)==ast.dump(expected,include_attributes=False)
    assert n['cells'][2]['metadata']['jupyter']['source_hidden']
    scope={};exec(''.join(n['cells'][1]['source']),scope)
    assert scope['REPO_URL']=='SET_YOUR_GITHUB_REPOSITORY_URL' and scope['USE_GOOGLE_DRIVE'] is True
    assert scope['REPO_REF']=='main' and scope['LOCAL_OUTPUT_DIR']=='/content/pde-inverse-outputs'
    scope.update(IS_COLAB=False)
    # Full bootstrap cell really executes its local branch; clone/install/mount guards elsewhere above.
    exec(code,scope)
    assert scope['BOOTSTRAP'] is None and not scope['SETUP']['persistence'].drive_enabled
    exec(''.join(n['cells'][4]['source']),scope)
    assert scope['RUN_MODE']=='preview'


def test_resolved_real_git_sha_and_versions_recorded_in_all_manifest_kinds(remote,monkeypatch):
    from pdeep import artifact_metadata
    from pdeep.diagnostics import source_identity
    from pdeep.data import Normalizers
    from pdeep.core import decode
    meta=read_manifest(existing_checkpoint());normalizers=decode(Normalizers,meta['normalization']['values'])
    # Exercise existing metadata writer against a real isolated Git source identity.
    original=source_identity(remote.source)
    assert original['git_revision']==remote.sha
    monkeypatch.setattr(artifact_metadata,'source_identity',lambda root:original)
    for kind in ('training_checkpoint','inference_export','diagnostic_bundle'):
        record=artifact_metadata.make_metadata(kind,'D',normalizers,run_id='provenance-fixture')
        assert record['source']['git_revision']==remote.sha
        assert record['runtime']['dependencies']['torch']!='NOT INSTALLED'


def test_simulated_drive_copy_complete_state_restore_without_updates(tmp_path):
    path=existing_checkpoint();mount=tmp_path/'simulated-drive';mount.mkdir()
    copied=sync_verified(path,PersistencePolicy(True,str(mount),True,'filesystem_test',drive_root=str(mount/'MyDrive/research')))
    context=initialize(ROOT,device='cpu',output_dir=tmp_path/'restored-local')
    staged=stage_resume_checkpoint(context,copied['Drive_manifest'],'B0')
    # Interop threads can only be set once by torch. Real resume belongs in a
    # fresh process, and must not change the surrounding test process settings.
    program=r'''import sys,torch
from pdeep.artifact_io import read_payload
from pdeep.artifact_rng import capture_rng
from pdeep.experiment_artifacts import load_resumable
from pdeep.artifact_tensors import tree_identity,state_identity
from pdeep.core import digest
path=sys.argv[1]
metadata,payload=read_payload(path)
engine=load_resumable(path,expected_architecture='B0',probe=False)
assert state_identity(engine.model)==metadata['component_inventory']['model_state_identity']
assert digest(tree_identity(engine.optimizer.state_dict()))==digest(tree_identity(payload['optimizer']))
assert engine.scheduler.state_dict()==payload['scheduler']
assert engine.state.successful==metadata['successful_updates'] and engine.state.attempted==payload['state']['attempted']
assert engine.state.sampler_cursor==payload['state']['sampler_cursor']
assert engine.sampler.state_dict(engine.state.sampler_cursor)==payload['sampler']
after=capture_rng(engine.rng_generators)
for key in ('python_random','numpy_global','registered'):
    assert digest(tree_identity(after[key]))==digest(tree_identity(payload['rng'][key]))
assert torch.equal(after['torch_CPU'],payload['rng']['torch_CPU'])
assert engine.state.persistence['last_verified_local_checkpoint']==path
print('Complete fresh-process portable resume verified; zero new updates.')
'''
    result=subprocess.run([sys.executable,'-c',program,str(staged)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert 'zero new updates' in result.stdout


def test_colab_without_drive_no_google_dependency(tmp_path,monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kw):
        if name.startswith('google'):pytest.fail('Drive-disabled setup imported Google')
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    result=setup.configure_storage(is_colab=True,use_drive=False,drive_root='/unused',local_output_dir=tmp_path/'local',project_dir=ROOT)
    assert (tmp_path/'local/training_runs').is_dir() and not result['persistence'].drive_enabled
