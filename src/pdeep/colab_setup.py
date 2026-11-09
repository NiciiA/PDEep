"""Credential-free Colab bootstrap and storage setup; no training operations.

The stdlib-only ensure_checkout function is embedded verbatim in the notebook's
collapsed setup cell because pdeep does not yet exist in a fresh Colab runtime.
No Git/pip stderr or environment contents are printed or written to reports.
"""
from pathlib import Path
import sys


def ensure_checkout(repo_url, repo_ref, project_dir, *, git_env=None):
    """Clone only absent destinations; existing worktrees are strictly read-only.

    Supports token-free HTTPS and Git-over-SSH URLs. External credential helpers
    or SSH agents handle private authentication. Raw Git output stays in memory.
    """
    import os, re, subprocess, tempfile
    from pathlib import Path
    from urllib.parse import urlsplit
    import sys
    if sys.version_info < (3, 10):
        raise RuntimeError('PDEep requires Python >=3.10; select a supported runtime before setup.')
    def identity(value):
        if not isinstance(value,str) or not value or any(c in value for c in '\n\r\0'):
            raise ValueError('Configure a token-free repository URL; credentials must not be embedded.')
        if value.startswith('git@'):
            match=re.fullmatch(r'git@([A-Za-z0-9.-]+):([A-Za-z0-9_./-]+)',value)
            if not match:raise ValueError('Use git@HOST:OWNER/REPO.git with an external SSH agent.')
            host,path=match.groups()
        else:
            url=urlsplit(value)
            if url.scheme not in {'https','ssh'} or not url.hostname or url.password or url.query or url.fragment:
                raise ValueError('Use token-free HTTPS or Git SSH; credential URLs, queries and fragments are rejected.')
            if (url.scheme=='https' and url.username is not None) or (url.scheme=='ssh' and url.username not in {None,'git'}):
                raise ValueError('Do not put credentials/user tokens in repository URLs; use an external credential helper or SSH agent.')
            host=url.hostname;path=url.path.lstrip('/')
            if url.port not in {None,22,443}:raise ValueError('Unsupported repository port.')
        path=path.rstrip('/').removesuffix('.git')
        if not path or len(path.split('/'))<2 or '..' in path.split('/'):
            raise ValueError('Configure REPO_URL with OWNER/REPO; no local or credential-bearing URL is accepted.')
        if host.lower()=='github.com':path=path.lower()
        return host.lower()+'/'+path
    expected=identity(repo_url)
    if not isinstance(repo_ref,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]*',repo_ref) or '..' in repo_ref:
        raise ValueError('REPO_REF must be a branch, tag or full commit SHA, not a command or revision expression.')
    destination=Path(project_dir).expanduser().resolve()
    env=dict(os.environ);env.update(git_env or {});env['GIT_TERMINAL_PROMPT']='0'
    def git(arguments, *, cwd=None, purpose='repository verification'):
        try:result=subprocess.run(['git',*arguments],cwd=cwd,env=env,capture_output=True,text=True,timeout=180)
        except FileNotFoundError:raise RuntimeError('Git is missing; install Git before repository setup.') from None
        except (OSError,subprocess.TimeoutExpired):raise RuntimeError('Git '+purpose+' could not complete; no credentials/output were logged.') from None
        if result.returncode:
            error=result.stderr.lower()
            if any(s in error for s in ('authentication','could not read username','publickey','permission denied')):
                raise RuntimeError('Git authentication failed. Configure an external credential helper or SSH agent with read access; do not put secrets in this notebook.')
            raise RuntimeError('Git '+purpose+' failed. Check repository existence, access and REPO_REF. Raw output is suppressed to protect credentials.')
        return result.stdout.strip()
    def head(path):return git(['rev-parse','--verify','HEAD^{commit}'],cwd=path)
    def resolved_ref(path):
        if re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}',repo_ref):wanted=repo_ref.lower()
        else:
            refs=[repo_ref,repo_ref+'^{}'] if repo_ref.startswith('refs/') else ['refs/heads/'+repo_ref,'refs/tags/'+repo_ref,'refs/tags/'+repo_ref+'^{}']
            lines=git(['ls-remote','--exit-code','origin',*refs],cwd=path,purpose='remote reference lookup')
            mapping=dict((ref,sha) for sha,ref in (line.split() for line in lines.splitlines()))
            if repo_ref.startswith('refs/'):wanted=mapping.get(repo_ref+'^{}',mapping.get(repo_ref))
            else:
                branch=mapping.get('refs/heads/'+repo_ref);tag=mapping.get('refs/tags/'+repo_ref+'^{}',mapping.get('refs/tags/'+repo_ref))
                if branch and tag and branch!=tag:raise ValueError('Branch/tag name is ambiguous; use refs/heads/... or refs/tags/....')
                wanted=branch or tag
            if not wanted:raise ValueError('Configured REPO_REF was not found on origin.')
        return wanted
    existed=destination.exists()
    if existed:
        if not destination.is_dir() or not (destination/'.git').exists():
            raise ValueError('PROJECT_DIR exists but is not a Git checkout; choose a separate empty directory. Nothing was overwritten.')
        top=Path(git(['rev-parse','--show-toplevel'],cwd=destination)).resolve()
        if top!=destination:raise ValueError('PROJECT_DIR must be the repository root, not a nested directory.')
        origin=git(['config','--get','remote.origin.url'],cwd=destination)
        if identity(origin)!=expected:raise ValueError('Existing checkout origin differs from REPO_URL. No files or remotes were changed.')
        if git(['status','--porcelain','--untracked-files=normal'],cwd=destination):
            raise ValueError('Existing checkout has local modifications/untracked files. Commit them yourself or choose a new PROJECT_DIR; no reset/checkout was performed.')
        commit=head(destination)
        wanted=resolved_ref(destination)
        if commit!=wanted:
            raise ValueError('Existing checkout is at '+commit+' but REPO_REF resolves to '+wanted+'. Choose a new PROJECT_DIR or update it explicitly; nothing was reset.')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        # A failed clone stays in its own scratch directory, not PROJECT_DIR.
        with tempfile.TemporaryDirectory(prefix='.pdeep-clone-',dir=destination.parent) as scratch:
            checkout=Path(scratch)/'checkout'
            git(['clone','--no-checkout','--',repo_url,str(checkout)],purpose='clone')
            wanted=resolved_ref(checkout)
            git(['fetch','--depth','1','origin',wanted],cwd=checkout,purpose='configured reference fetch')
            git(['checkout','--detach','FETCH_HEAD'],cwd=checkout,purpose='fresh checkout')
            commit=head(checkout)
            if commit!=wanted:raise RuntimeError('Fetched checkout differs from the resolved reference; setup stopped.')
            required=('pyproject.toml','src/pdeep/__init__.py','src/pdeep/colab_setup.py','docs/scientific_contract_v1.md')
            if any(not (checkout/p).is_file() for p in required):
                raise ValueError('Repository/ref lacks the expected PDEep structure or this Colab setup version; select a compatible commit.')
            os.rename(checkout,destination)
    required=('pyproject.toml','src/pdeep/__init__.py','src/pdeep/colab_setup.py','docs/scientific_contract_v1.md')
    if any(not (destination/p).is_file() for p in required):raise ValueError('Expected pyproject.toml, pdeep package/setup and scientific contract are missing.')
    return {'project_dir':str(destination),'repository_identity':expected,'requested_ref':repo_ref,'git_commit':commit,
            'existing_checkout':existed,'worktree_modified_by_setup':not existed,'credentials_recorded':False}


def running_in_colab(shell=None):
    """No google package or Drive import on local machines."""
    if shell is None:
        try:
            from IPython import get_ipython
            shell=get_ipython()
        except ImportError:pass
    return 'google.colab' in sys.modules or type(shell).__module__.startswith('google.colab')


def install_checkout(project_dir,*,runner=None):
    """Use pyproject editable installation; suppress all potentially secret output."""
    import subprocess,importlib,site
    root=Path(project_dir).resolve()
    if sys.version_info<(3,10):raise RuntimeError('Unsupported Python: PDEep requires >=3.10.')
    text=(root/'pyproject.toml').read_text()
    if 'requires-python = ">=3.10"' not in text:
        raise ValueError('Project Python requirement changed; inspect it before automatic installation.')
    if 'pdeep' in sys.modules:
        actual=Path(sys.modules['pdeep'].__file__).resolve().parents[2]
        if actual!=root:raise ValueError('Another pdeep checkout is already imported; restart the runtime before installing this checkout.')
    runner=runner or subprocess.run
    try:result=runner([sys.executable,'-m','pip','install','-e',str(root)],capture_output=True,text=True,timeout=600)
    except (OSError,subprocess.TimeoutExpired):raise RuntimeError('Editable installation could not finish. Check runtime/network and Python support; raw logs suppressed.') from None
    if result.returncode:raise RuntimeError('Editable installation failed. Check pyproject/Python compatibility and dependency access; raw installer output is not logged.')
    importlib.invalidate_caches()
    for folder in site.getsitepackages():site.addsitedir(folder)
    site.addsitedir(site.getusersitepackages())
    return verify_import(root)


def verify_import(project_dir):
    import importlib
    pdeep=importlib.import_module('pdeep');expected=Path(project_dir).resolve()/'src/pdeep'
    actual=Path(pdeep.__file__).resolve().parent
    if actual!=expected:raise RuntimeError('pdeep imports from another source directory; restart the kernel and install the configured checkout.')
    return str(actual)


def validate_cuda():
    import torch
    from pdeep.runtime import resolve_device
    try:
        device,dtype=resolve_device('cuda','float32')
        value=torch.ones(1,device=device,dtype=dtype).sum().item()
        if value!=1:raise RuntimeError('CUDA probe result invalid')
        return {'device':'cuda','precision':'float32','name':torch.cuda.get_device_name(device),'count':torch.cuda.device_count(),
                'compute_capability':list(torch.cuda.get_device_capability(device)),'probe_passed':True}
    except (ValueError,RuntimeError,AssertionError) as error:
        raise RuntimeError('Colab requires working CUDA for this workflow. Select a GPU runtime and rerun setup; no CPU fallback occurred.') from None


def configure_storage(*,is_colab,use_drive,drive_root,local_output_dir,project_dir,mount_point='/content/drive'):
    import tempfile,os
    from pdeep.artifact_drive import PersistencePolicy
    if not is_colab:
        return {'local_output_dir':str(Path(project_dir)/'artifacts'),'drive_root':None,'drive_mount':None,
                'persistence':PersistencePolicy(),'Drive_destination_ready':False,'message':'Local session: no clone, install or Google Drive import/mount; local outputs only.'}
    local=Path(local_output_dir).expanduser().resolve();mount=Path(mount_point).resolve()
    if local==mount or local.is_relative_to(mount):raise ValueError('LOCAL_OUTPUT_DIR must be local Colab storage, outside the Drive mount.')
    local.mkdir(parents=True,exist_ok=True)
    for name in ('training_runs','setup_reports'):(local/name).mkdir(exist_ok=True)
    with tempfile.TemporaryFile(dir=local) as stream:stream.write(b'local-write-check');stream.flush()
    policy=PersistencePolicy();root=None
    if use_drive:
        root=Path(drive_root).expanduser().resolve()
        if not root.is_relative_to(mount):raise ValueError('DRIVE_ROOT must be inside the configured Google Drive mount.')
        from google.colab import drive
        drive.mount(str(mount))
        if not os.path.ismount(mount):raise RuntimeError('Google Drive API returned without a mounted filesystem; persistence is not enabled.')
        for name in ('checkpoints','inference','reports'):(root/name).mkdir(parents=True,exist_ok=True)
        try:
            with tempfile.TemporaryFile(dir=root) as stream:stream.write(b'drive-write-check');stream.flush()
        except OSError:raise RuntimeError('DRIVE_ROOT is not writable; fix its permissions before training.') from None
        policy=PersistencePolicy(drive_enabled=True,drive_mount=str(mount),drive_explicitly_mounted=True,drive_root=str(root))
    return {'local_output_dir':str(local),'drive_root':None if root is None else str(root),'drive_mount':str(mount) if use_drive else None,
            'persistence':policy,'Drive_destination_ready':bool(use_drive),
            'message':'Drive destination mounted/writable; checkpoint verification occurs only after each successful copy.' if use_drive else 'Colab local output only: runtime storage is temporary; Drive disabled.'}


def finish_setup(*,is_colab,bootstrap=None,use_drive=True,drive_root='/content/drive/MyDrive/PDEInverseResearch',
                 local_output_dir='/content/pde-inverse-outputs',mount_point='/content/drive'):
    import pdeep
    from pdeep.runtime import environment_report
    from pdeep.diagnostics import source_identity
    root=Path(pdeep.__file__).resolve().parents[2]
    if is_colab:
        if bootstrap is None or root!=Path(bootstrap['project_dir']).resolve():raise RuntimeError('Configured checkout/import mismatch.')
        verify_import(root);cuda=validate_cuda()
    else:cuda=None
    storage=configure_storage(is_colab=is_colab,use_drive=use_drive,drive_root=drive_root,local_output_dir=local_output_dir,project_dir=root,mount_point=mount_point)
    source=source_identity(root)
    if is_colab and source['git_revision']!=bootstrap['git_commit']:raise RuntimeError('Git revision changed after bootstrap; rerun setup before training.')
    result={**storage,'is_colab':is_colab,'project_dir':str(root),'git_commit':source['git_revision'],
            'source_identity':source['sha256'],'runtime':environment_report(),'cuda':cuda,'bootstrap':bootstrap,
            'training_device':'cuda' if is_colab else 'auto','credentials_recorded':False}
    if is_colab:
        from dataclasses import asdict
        from pdeep.artifact_io import atomic_json
        atomic_json(Path(storage['local_output_dir'])/'setup_reports/setup.json',{**result,'persistence':asdict(result['persistence'])})
    return result
