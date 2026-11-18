"""Conservative host/cgroup-aware process and full-state memory budgets."""
import os,math
from pathlib import Path
import psutil

def budget(certificate,families,*,requested=0,block_cases=2,available_bytes=None,needs_finer=True):
    cpu=getattr(os,'process_cpu_count',os.cpu_count)() or 1
    if hasattr(os,'sched_getaffinity'):cpu=min(cpu,len(os.sched_getaffinity(0)))
    available=psutil.virtual_memory().available if available_bytes is None else available_bytes
    quota=Path('/sys/fs/cgroup/cpu.max');limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
    if quota.exists():
        q,p=quota.read_text().split()
        if q!='max':cpu=min(cpu,max(1,math.ceil(int(q)/int(p))))
    if limit.exists() and current.exists() and limit.read_text().strip()!='max':available=min(available,max(0,int(limit.read_text())-int(current.read_text())))
    nodes=max(certificate['mesh_records'][g]['production']['free_DOFs'] for g in families)
    # Both full501-state arrays + all sparse factors/assembly/host Python allowance.
    per_worker=int(640*1024**2+nodes*(2*501*8*block_cases+12000+(8000 if needs_finer else 0)));reserve=max(1024**3,int(.15*available))
    # Coordinator is primarily IO; reserving an entire core would unnecessarily
    # force single-worker execution on the common two-vCPU Colab runtime.
    safe=max(1,min(cpu,len(families),(max(0,available-reserve)//per_worker)))
    selected=safe if requested==0 else requested
    if selected<1 or selected>safe:raise ValueError(f'requested{selected} workers exceeds safe{safe}: availableRAM={available/1e9:.2f}GB, estimated worker={per_worker/1e9:.2f}GB, reserve={reserve/1e9:.2f}GB; reduce workers/block cases')
    if available-reserve<per_worker:raise ValueError('insufficient RAM for even one declared full-reference worker; no coarsening/fallback')
    return {'available_CPUs':cpu,'available_RAM_bytes':available,'reserve_bytes':reserve,'estimated_worker_peak_bytes':per_worker,'safe_max_workers':safe,'workers':selected,'block_cases':block_cases,'threads_per_worker':1,'independent_finer_workspace_included':needs_finer,'memory_scope':'conservative estimate; actual aggregate RSS sampled during execution'}
