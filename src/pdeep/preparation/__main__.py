"""python -m pdeep.preparation — ordinary-filesystem portable dataset tools."""
import argparse,json
from pathlib import Path
from .portable import verify_folder,read_index
from .transfer import sync_folder,archive_dataset,extract_archive,export_legacy

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output');a.add_argument('--manifest');a.add_argument('--certificate');a.add_argument('--preset',choices=('smoke','pilot','research'));a.add_argument('--profile',choices=('feasibility','research'),default='research');a.add_argument('--updates',type=int);a.add_argument('--queries',type=int);a.add_argument('--sync-to');a.add_argument('--max-cases',type=int);a.add_argument('--workers',type=int,default=0);a.add_argument('--block-cases',type=int,default=2);a.add_argument('--reuse-from')
    a=sub.add_parser('status');a.add_argument('source');a.add_argument('--verify',choices=('index','checksums','payloads'),default='index');a.add_argument('--mirror')
    a=sub.add_parser('prepare-label');a.add_argument('--manifest',required=True);a.add_argument('--request',required=True);a.add_argument('--certificate')
    a=sub.add_parser('prepare-label-batch');a.add_argument('--manifest',required=True);a.add_argument('--audit',required=True);a.add_argument('--certificate')
    a=sub.add_parser('verify');a.add_argument('source');a.add_argument('--complete',action='store_true')
    a=sub.add_parser('sync');a.add_argument('source');a.add_argument('destination')
    a=sub.add_parser('restore');a.add_argument('source');a.add_argument('destination')
    a=sub.add_parser('pack');a.add_argument('source');a.add_argument('archive')
    a=sub.add_parser('export-legacy');a.add_argument('source');a.add_argument('destination');a.add_argument('--certificate',required=True,help='exact full producer certificate JSON; no new qualification')
    args=p.parse_args(argv)
    if args.command=='prepare-label-batch':
        from ..learning_diagnostics.certificates import resolve_certification
        from .supplements import prepare_label_batch
        result=prepare_label_batch(args.manifest,args.audit,resolve_certification(requested_capabilities={'certification_id':None},configured_path=args.certificate).require())
    elif args.command=='prepare-label':
        from ..learning_diagnostics.certificates import resolve_certification
        from .supplements import prepare_label
        result=prepare_label(args.manifest,args.request,resolve_certification(requested_capabilities={'certification_id':None},configured_path=args.certificate).require())
    elif args.command=='status':
        from .status import status
        result=status(args.source,verify=args.verify,mirror=args.mirror)
    elif args.command=='verify':
        path,index=verify_folder(args.source,require_complete=args.complete);result={'manifest':str(path),'status':index['status'],'dataset_identity':index['dataset_identity'],'verified_files':len(index['files']),'prepared_cases':len(index['records'])}
    elif args.command=='sync':result=sync_folder(args.source,args.destination,progress=lambda s,**k:print(s,k,flush=True))
    elif args.command=='restore':result=extract_archive(args.source,args.destination) if args.source.endswith('.tar.gz') else sync_folder(args.source,args.destination)
    elif args.command=='pack':result=archive_dataset(args.source,args.archive)
    elif args.command=='export-legacy':
        from ..smooth.certification import load_record
        result={'manifest':str(export_legacy(args.source,args.destination,load_record(args.certificate,current=False)))}
    else:
        from ..learning_diagnostics.certificates import resolve_certification
        from ..unified.config import DatasetConfig
        from .parallel import prepare_dataset
        cert=resolve_certification(requested_capabilities={'certification_id':None},configured_path=args.certificate).require()
        if args.manifest:
            path,index=read_index(args.manifest);config=DatasetConfig(**index['config']);folder=path.parent
            if args.output and Path(args.output).expanduser().resolve()!=folder.resolve():p.error('resume output must be the selected manifest folder; use restore to move it first')
            for value,expected in ((args.preset,config.preset),(args.updates,config.stream_updates),(args.queries,config.stream_queries)):
                if value is not None and value!=expected:p.error('explicit resume settings differ from the selected manifest')
        else:
            if not args.output:p.error('fresh preparation requires --output')
            if args.profile=='feasibility':
                from .feasibility import configuration
                config=configuration()
                if args.updates not in (None,3000) or args.queries not in (None,4096):p.error('feasibility declares3000 updates and4096 queries; no silent scientific reductions')
            else:
                preset=args.preset or 'research';updates=args.updates if args.updates is not None else (100000 if preset=='research' else None)
                config=DatasetConfig(preset=preset,stream_updates=updates,stream_queries=args.queries)
            folder=Path(args.output).expanduser()
        result=prepare_dataset(config,folder,cert,manifest=args.manifest,sync_to=args.sync_to,max_cases=args.max_cases,workers=args.workers,block_cases=args.block_cases,feasibility=args.profile=='feasibility',reuse_from=args.reuse_from).summary()
    print(json.dumps(result,indent=2,default=str));return 0

if __name__=='__main__':main()
