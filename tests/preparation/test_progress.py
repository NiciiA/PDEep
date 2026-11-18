from pdeep.preparation.workflow import progress_callback
from pdeep.unified_modes.dashboard import Dashboard

def test_measured_task_eta_visible_without_invented_whole_study_eta():
    monitor=Dashboard(notebook=False);report=progress_callback(monitor)
    report('Dataset inspection',task='wave/hole',prepared=4,required=224,cases_per_second=.1,task_ETA_seconds=2200.,whole_dataset_ETA_seconds=None)
    assert monitor.latest['preparation_task']=='wave/hole'
    assert monitor.latest['task_ETA_seconds']==2200. and monitor.latest['task_cases_per_second']==.1
    assert monitor.latest['whole_study_ETA_seconds'] is None
    assert 'unknown' in monitor.latest['ETA_status']
    report('Dataset case verified',task='wave/hole',task_ETA_seconds=2000.,whole_dataset_ETA_seconds=5000.,cases_per_second=.11)
    assert monitor.latest['whole_study_ETA_seconds']==5000.

def test_notebook_generation_rejects_drive_as_working_directory(certificate,tmp_path):
    import pytest
    from pdeep.artifact_drive import PersistencePolicy
    from pdeep.unified.config import DatasetConfig
    from pdeep.preparation.workflow import prepare
    mount=tmp_path/'drive';mount.mkdir()
    policy=PersistencePolicy(drive_enabled=True,drive_mount=str(mount),drive_explicitly_mounted=True,backend='filesystem_test',drive_root=str(mount/'project'))
    with pytest.raises(ValueError,match='local storage'):
        prepare(DatasetConfig(preset='smoke'),mount/'G1_unified',certificate,action='prepare',persistence=policy,monitor=Dashboard(notebook=False))

def test_cli_smoke_budget_and_research_budget_are_explicit(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    from pdeep.preparation.__main__ import main
    seen=[]
    monkeypatch.setattr('pdeep.learning_diagnostics.certificates.resolve_certification',lambda **kw:SimpleNamespace(require=lambda:{}))
    def no_generation(config,*args,**kwargs):
        seen.append(config);return SimpleNamespace(summary=lambda:{'scope':'configuration-only unit control'})
    monkeypatch.setattr('pdeep.preparation.parallel.prepare_dataset',no_generation)
    assert main(['prepare','--preset','smoke','--output',str(tmp_path/'smoke')])==0
    assert (seen[-1].stream_updates,seen[-1].stream_queries)==(3,128)
    assert main(['prepare','--output',str(tmp_path/'research')])==0
    assert (seen[-1].stream_updates,seen[-1].stream_queries)==(100000,4096)

def test_feasibility_uses_exact_local_parent_and_blocks_legacy_regeneration(tmp_path,monkeypatch):
    import pytest
    from pdeep.preparation.feasibility import configuration,parent_configuration
    from pdeep.preparation.workflow import prepare
    c=configuration();parent=tmp_path/'datasets'/parent_configuration(c).identity;parent.mkdir(parents=True)
    source=parent/'manifest.json';source.write_text('{}');seen={}
    def capture(*a,**kwargs):seen.update(kwargs);return 'captured'
    monkeypatch.setattr('pdeep.preparation.parallel.prepare_dataset',capture)
    assert prepare(c,tmp_path,None,action='prepare',feasibility=True)=='captured'
    assert seen['reuse_from']==source
    source.rename(parent/'index.json')
    with pytest.raises(ValueError,match='no completed cases regenerated'):
        prepare(c,tmp_path,None,action='prepare',feasibility=True)
