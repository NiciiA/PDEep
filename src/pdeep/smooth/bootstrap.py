"""Optional Colab reference-environment preparation, stdlib only.

Never runs locally and never changes an already imported numerical stack.
Call via runpy after repository verification and before finish_setup imports.
"""
from pathlib import Path
import importlib.metadata
import json,sys,subprocess


def prepare_reference_environment(project_dir,*,is_colab,runner=None):
    root=Path(project_dir).resolve();base=root/'certifications/G1_smooth_v1'
    if not is_colab:return {'status':'local installed environment unchanged','installed':False}
    if sys.version_info<(3,12):raise RuntimeError('This G1_smooth_v1 numerical certificate requires Python >=3.12 (pinned NumPy/SciPy). Select a compatible Colab runtime; do not bypass certification.')
    pointer=base/'default.json'
    if not pointer.is_file():raise RuntimeError('This repository/ref has no G1_smooth_v1 certification. Push the certification JSON and source files, then select that commit.')
    pointer=json.loads(pointer.read_text());record=json.loads((base/pointer['record']).read_text());wanted=record['source_revision']['dependencies']
    actual={}
    for name in wanted:
        try:actual[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual[name]=None
    changed=[name for name in wanted if actual[name]!=wanted[name]]
    if not changed:return {'status':'pinned reference dependencies already installed','installed':False,'versions':wanted}
    imported=[name for name in changed if name in sys.modules]
    runner=runner or subprocess.run
    result=runner([sys.executable,'-m','pip','install','-r',str(base/'requirements.txt')],capture_output=True,text=True,timeout=600)
    if result.returncode:raise RuntimeError('Certified reference dependency installation failed. Check supported Python/package availability; installer output is suppressed to protect credentials. Do not train with a mismatched certificate.')
    actual={name:importlib.metadata.version(name) for name in wanted}
    if imported:raise RuntimeError('Certified reference dependencies are now installed, but this Colab kernel had already imported older versions: '+', '.join(imported)+'. Restart the runtime once, then rerun setup. No old modules were replaced in place; training remains blocked until restart.')
    if actual!=wanted:raise RuntimeError('Pinned reference dependency versions did not resolve. Restart the runtime and inspect the selected repository certification.')
    return {'status':'installed pinned certified reference dependencies before import','installed':True,'versions':actual}
