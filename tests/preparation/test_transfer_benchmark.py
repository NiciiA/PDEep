"""Local filesystem benchmark: actual small dataset plus2000 tiny diagnostics."""
from pathlib import Path
import json,time
from pdeep.artifact_io import atomic_json,sha256
from pdeep.preparation.portable import read_index,commit,add_file
from pdeep.preparation.transfer import sync_folder,archive_dataset,extract_archive,copy_file

def test_bounded_folder_vs_archive_benchmark(prepared,tmp_path):
    source=tmp_path/'source';sync_folder(prepared.path,source);path,index=read_index(source)
    # Transfer-only control, clearly separate from physical labels and QC.
    for i in range(2000):
        p=source/'quality_reports/transfer-only-control'/f'{i:04d}.json';atomic_json(p,{'scope':'transfer benchmark auxiliary control; not numerical qualification','index':i,'text':'independent checksum-verified diagnostic record'})
        add_file(index,source,p)
    commit(path,index)
    a=time.perf_counter();folder=sync_folder(path,tmp_path/'folder-copy');folder_s=time.perf_counter()-a
    archive=tmp_path/'transfer.tar.gz';packed=archive_dataset(path,archive)
    sent=tmp_path/'archive-copy/transfer.tar.gz';a=time.perf_counter();copy_file(archive,sent,{'sha256':sha256(archive),'size_bytes':archive.stat().st_size});copy_file(Path(str(archive)+'.json'),Path(str(sent)+'.json'),{'sha256':sha256(str(archive)+'.json'),'size_bytes':Path(str(archive)+'.json').stat().st_size});copy_s=time.perf_counter()-a
    extracted=extract_archive(sent,tmp_path/'archive-restored')
    result={'scope':'local macOS filesystem only;8 actual P4 Poisson cases plus2000 auxiliary tiny files; no mounted Drive/network performance claim','files':len(index['files'])+1,'uncompressed_bytes':sum(v['size_bytes'] for v in index['files'].values()),'ordinary_verified_folder_seconds':folder_s,'archive_creation_seconds':packed['seconds'],'verified_archive_copy_seconds':copy_s,'archive_extract_verify_seconds':extracted['seconds'],'archive_end_to_end_seconds':packed['seconds']+copy_s+extracted['seconds'],'archive_bytes':packed['size_bytes'],'folder_to_archive_end_to_end_ratio':folder_s/(packed['seconds']+copy_s+extracted['seconds']),'persistent_archives_left':0}
    out=Path('reports/g1_preparation');out.mkdir(parents=True,exist_ok=True);(out/'transfer_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
