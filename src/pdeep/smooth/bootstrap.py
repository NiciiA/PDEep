"""Stdlib-only, idempotent preparation BEFORE project/scientific imports.

Keep a consistent preloaded Colab numerical stack. Its numerical validity is
established by explicit bounded qualification, never by loosening version checks.
Never pip-replace an imported binary. No reload, sys.modules removal or restart.
"""
from pathlib import Path
import importlib.metadata
import json,sys,subprocess,time

MINIMUM={'numpy':(1,26),'scipy':(1,11),'triangle':(20250106,),'shapely':(2,0),'torch':(2,2),'matplotlib':(3,8)}

def version_tuple(value):
    import re
    return tuple(int(v) for v in re.match(r'\d+(?:\.\d+)*',value).group().split('.'))

def environment_versions():
    result={}
    for name in MINIMUM:
        try:installed=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:installed=None
        module=sys.modules.get(name);loaded=None if module is None else str(getattr(module,'__version__','unknown'))
        result[name]={'installed':installed,'loaded':loaded}
    return result

def run_install(argv,*,runner=None,progress=print):
    """No raw installer output/credentials; emit a heartbeat during slow installs."""
    if runner is not None:
        result=runner(argv,capture_output=True,text=True,timeout=600)
        if result.returncode:raise RuntimeError('Dependency/project installation failed; raw installer logs suppressed. Check package access and supported Python.')
        return
    start=time.monotonic();process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        while True:
            try:process.communicate(timeout=15);break
            except subprocess.TimeoutExpired:
                progress(f'Installation still running ({time.monotonic()-start:.0f}s); no scientific imports or optimizer updates.')
                if time.monotonic()-start>600:raise RuntimeError('Installation timed out; no scientific imports were attempted.')
        if process.returncode:raise RuntimeError('Dependency/project installation failed; raw installer logs suppressed. Check package access and Python compatibility.')
    finally:
        if process.poll() is None:
            import os,signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.communicate()

def prepare_reference_environment(project_dir,*,is_colab,runner=None,progress=print):
    if not is_colab:return {'status':'local installed environment unchanged','installed':False,'python':sys.executable,'versions':environment_versions()}
    if sys.version_info<(3,12):raise RuntimeError('This certified smooth workflow requires Python >=3.12. Select a current Colab runtime; no kernel restart/install was attempted.')
    root=Path(project_dir).resolve();base=root/'certifications/G1_smooth_v1';pointer=base/'default.json'
    if not pointer.is_file():raise RuntimeError('Repository/ref lacks the smooth certificate; select a complete project revision.')
    record=json.loads((base/json.loads(pointer.read_text())['record']).read_text());fallback=record['source_revision']['dependencies']
    before=environment_versions();progress('Bootstrap interpreter: '+sys.executable);progress('Installed / already loaded numerical versions: '+json.dumps(before,sort_keys=True))
    for name,entry in before.items():
        if entry['loaded'] is not None and entry['loaded']!=entry['installed']:
            raise RuntimeError(f'Contaminated session: {name} loaded={entry["loaded"]}, installed={entry["installed"]}. A previous installer already replaced a loaded binary. This kernel cannot safely continue. Use a NEW Colab runtime with the corrected notebook. Nothing was installed/reloaded/restarted by this setup.')
        version=entry['installed']
        if version is not None and version_tuple(version)<MINIMUM[name]:
            raise RuntimeError(f'Unsupported installed {name}={version}; minimum API version is {MINIMUM[name]}. Select a current runtime. Loaded scientific binaries will not be replaced.')
    changed=[]
    for name in ('numpy','scipy','triangle','shapely'):
        if before[name]['installed'] is None:
            wanted=fallback[name];progress(f'Installing missing {name}=={wanted} with --no-deps; retaining existing numerical packages.')
            run_install([sys.executable,'-m','pip','install','--no-deps',name+'=='+wanted],runner=runner,progress=progress);changed.append(name)
    for name in ('torch','matplotlib'):
        if before[name]['installed'] is None:raise RuntimeError(f'Colab runtime is missing {name}; select a supported GPU runtime. No replacement CUDA/Torch stack was installed.')
    progress('Retaining installed NumPy/SciPy/Torch versions. Reference validity will be checked explicitly before plan preparation.')
    return {'status':'consistent environment prepared; numerical qualification required before training','installed':bool(changed),'packages_installed':changed,'python':sys.executable,'versions':environment_versions(),'restart_required':False,'qualification_required':True}

def install_project(project_dir,*,runner=None,progress=print,source_revision=None):
    root=Path(project_dir).resolve();marker=root/'.pdeep-editable-setup.json'
    # Only source metadata/configuration, no import of pdeep/scientific packages.
    import hashlib
    key={'python':sys.executable,'source_revision':source_revision,'pyproject_sha256':hashlib.sha256((root/'pyproject.toml').read_bytes()).hexdigest()}
    if marker.is_file() and json.loads(marker.read_text())!=key and 'pdeep' in sys.modules:
        raise RuntimeError('Project revision changed after import. Use a new runtime for the selected commit; source modules will not be reloaded in place.')
    if marker.is_file() and json.loads(marker.read_text())==key:
        progress('Editable project installation already prepared; no repeated pip call.');return
    progress('Installing project editable with --no-deps; notebook kernel and numerical binaries stay unchanged.')
    run_install([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(root)],runner=runner,progress=progress)
    import site,importlib
    for folder in site.getsitepackages():site.addsitedir(folder)
    site.addsitedir(site.getusersitepackages());importlib.invalidate_caches()
    marker.write_text(json.dumps(key)+'\n')
