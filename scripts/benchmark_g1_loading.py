"""Bounded complete288-case LOAD benchmark, never generation or training.

Cold means a fresh Python process; OS cache is NOT purged. Before uses a pinned
old loader. Restore measures ordinary local copying, not Drive/network. Optional
--profile adds cProfile overhead; default latency measurements omit it.
"""
import time
IMPORT_START=time.perf_counter()
import argparse,json,resource,sys,subprocess,tempfile,cProfile,pstats
from pathlib import Path
import torch
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.smooth.reference import Reference
from pdeep.wave_fidelity.reference import Space
from pdeep.geometric.fem import FEM
from pdeep.smooth.data import Case

def main():
    imports=time.perf_counter()-IMPORT_START
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('before','after','restore'));parser.add_argument('--manifest',required=True)
    parser.add_argument('--profile',action='store_true');parser.add_argument('--baseline-ref',default='5deab4c44c86ac33284486140f016624f516060e')
    args=parser.parse_args();rows=[]
    def stage(name,fn):
        start=time.perf_counter();value=fn();rows.append({'stage':name,'seconds':time.perf_counter()-start});return value
    def forbidden(*a,**kw):raise AssertionError('ordinary loading attempted FEM/reference computation')
    Reference.__init__=forbidden;Space.__init__=forbidden;FEM.__init__=forbidden;Case.evaluate=forbidden;torch.set_num_threads(1)
    cert=stage('existing_certificate_resolution',lambda:resolve_certification(requested_capabilities={'certification_id':None}).require());path=Path(args.manifest)
    from pdeep.preparation.portable import read_index
    assert len(read_index(path)[1]['records'])==288,'bounded benchmark requires the complete288-case dataset'
    profiler=cProfile.Profile()
    if args.profile:profiler.enable()
    if args.mode=='restore':
        from pdeep.preparation.workflow import prepare
        from pdeep.unified.config import DatasetConfig
        from pdeep.unified_modes.dashboard import Dashboard
        config=DatasetConfig(**read_index(path)[1]['config'])
        with tempfile.TemporaryDirectory(prefix='g1-local-restore-benchmark-') as root:
            call=lambda:prepare(config,root,cert,action='restore',manifest=path,monitor=Dashboard(config,notebook=False))
            d=stage('first_local_copy_verify_load',call);first=dict(d.load_profile['restore'])
            e=stage('repeat_restore_reuses_local',call)
            restore={'first':first,'repeat':e.load_profile['restore'],'Drive_transfer_seconds':None,'scope':'local filesystem only; actual Drive not mounted'}
    else:
        if args.mode=='before':
            code=subprocess.run(['git','show',args.baseline_ref+':src/pdeep/preparation/portable.py'],check=True,capture_output=True,text=True).stdout
            namespace={'__name__':'pdeep.preparation._baseline_loading','__package__':'pdeep.preparation'}
            exec(compile(code,'baseline_portable.py','exec'),namespace);load=lambda:namespace['Dataset'](path,cert)
        else:
            from pdeep.preparation.loading import load_dataset
            load=lambda:load_dataset(path,cert)
        d=stage('cold_process_constructor',load)
        for split in ('train','validation','test'):stage('cold_readiness_'+split,lambda:d.require_ready(split))
        for split in ('train','validation','test'):stage('warm_readiness_'+split,lambda:d.require_ready(split))
        e=stage('repeat_notebook_load',load)
        for split in ('train','validation','test'):stage('explicit_all_payloads_'+split,lambda:[e.require_case(c) for c in e.cases(split)])
        restore=None
    if args.profile:profiler.disable()
    out=Path('reports/g1_preflight_loading');out.mkdir(parents=True,exist_ok=True)
    if args.profile:
        with (out/(args.mode+'-profile.txt')).open('w') as f:pstats.Stats(profiler,stream=f).sort_stats('cumtime').print_stats(50)
    result={'scope':'complete288 cases; cold process, OS cache not purged; local SSD','mode':args.mode,'cProfile_enabled':args.profile,'Python_project_import_seconds':imports,'stages':rows,'cases':288,'peak_host_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'FEM_assemblies':0,'factorizations':0,'PDE_solves':0,'continuous_reference_evaluations':0,'multiprocessing_workers':0,'load_profile':getattr(e,'load_profile',{}),'resident_label_bytes':getattr(e.cache,'resident_bytes',None),'same_dataset_object':e is d,'transfer':restore}
    name=args.mode+('-profiled' if args.profile else '-timing')+'.json';(out/name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
