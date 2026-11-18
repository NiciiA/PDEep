"""Simple notebook facade; preview is inert and restored data stay on local disk."""
from pathlib import Path
from ..core import require
from ..artifact_drive import PersistencePolicy
from ..unified_modes.dashboard import Dashboard
from .portable import Dataset,resolve_manifest,read_index
from .transfer import sync_folder,extract_archive

def progress_callback(dashboard):
    def report(stage,**details):
        if 'task_rows' in details:
            dashboard.latest.update(dataset_tasks=details['task_rows'],verified_physical_cases=details['physical_cases_total'],target_physical_cases=details['target_total'],active_reference_workers=details['active_workers'],configured_reference_workers=details['configured_workers'],preparation_cases_per_second=details['cases_per_second'],peak_host_RSS_bytes=details['peak_aggregate_RSS_bytes'],failed_reference_cases=details['failed_cases'],whole_dataset_ETA_seconds=details.get('whole_dataset_ETA_seconds'),whole_study_ETA_seconds=details.get('whole_dataset_ETA_seconds'),ETA_status='measured concurrent per-task costs; unmeasured tasks remain unknown')
        if 'task' in details:dashboard.latest['preparation_task']=details['task']
        if 'task_ETA_seconds' in details:
            whole=details.get('whole_dataset_ETA_seconds')
            dashboard.latest.update(task_ETA_seconds=details['task_ETA_seconds'],task_cases_per_second=details.get('cases_per_second'),whole_dataset_ETA_seconds=whole,
                whole_study_ETA_seconds=whole,ETA_status='measured per-task; whole dataset unknown until every task measured' if whole is None else 'measured per-task rolling medians, including writes/sync')
        return dashboard.phase(stage,**details)
    return report

def prepare(config,root,certificate,*,action='preview',manifest=None,persistence=None,max_cases=None,monitor=None,workers=0,block_cases=2,feasibility=False,reuse_from=None):
    require(action in ('preview','prepare','load','restore'),'invalid dataset action')
    if action=='preview':return None
    dashboard=monitor or Dashboard(config);report=progress_callback(dashboard);root=Path(root);folder=root/'datasets'/config.identity
    with dashboard.live():
        if action=='restore':
            require(manifest,'restore requires exact Drive dataset directory, manifest, or archive')
            dashboard.phase('Restoring verified dataset to fast local storage',source=str(manifest))
            if str(manifest).endswith('.tar.gz'):
                result=extract_archive(manifest,folder)
            else:
                _,source=read_index(manifest);require(source['dataset_identity']==config.identity,'selected dataset differs from configured plan')
                result=sync_folder(manifest,folder,progress=report)
            manifest=result['manifest']
        if action in ('load','restore'):
            require(manifest,'load requires an exact portable dataset manifest/directory')
            policy=persistence or PersistencePolicy()
            selected=resolve_manifest(manifest)
            require(policy.drive_mount is None or not selected.resolve().is_relative_to(Path(policy.drive_mount).resolve()),'use restore to copy the selected Drive dataset to fast local storage before loading')
            data=Dataset(manifest,certificate,progress=report)
            require(data.config.identity==config.identity,'loaded dataset/configuration differs')
            return data
        if manifest:
            selected,index=read_index(manifest);require(index['dataset_identity']==config.identity,'resume dataset/configuration differs');folder=selected.parent;manifest=selected
        persistence=persistence or PersistencePolicy();sync_to=None
        require(persistence.drive_mount is None or not folder.resolve().is_relative_to(Path(persistence.drive_mount).resolve()),'restore the dataset to local storage before continuing generation; local verification precedes Drive sync')
        if persistence.drive_enabled:sync_to=Path(persistence.drive_root or persistence.drive_mount)/'G1_unified/datasets'/config.identity
        if feasibility and reuse_from is None:
            # Exactly one configured parent location, never a directory/Drive
            # scan. Avoid accidentally solving its completed cases a second time.
            from .feasibility import parent_configuration
            parent=root/'datasets'/parent_configuration(config).identity
            if (parent/'manifest.json').is_file():
                reuse_from=parent/'manifest.json';report('Reusing exact local research-parent manifest',manifest=str(reuse_from))
            elif (parent/'index.json').is_file():
                raise ValueError('completed legacy research data found at '+str(parent/'index.json')+'; export that exact index to a portable manifest and set reuse_from before preparing feasibility; no completed cases regenerated')
        # Preview/load never initialize multiprocessing infrastructure. Worker
        # modules are needed only for explicitly requested offline generation.
        from .parallel import prepare_dataset
        return prepare_dataset(config,folder,certificate,manifest=manifest,progress=report,sync_to=sync_to,max_cases=max_cases,workers=workers,block_cases=block_cases,feasibility=feasibility,reuse_from=reuse_from)
