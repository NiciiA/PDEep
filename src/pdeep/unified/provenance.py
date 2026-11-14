"""Execution-cache/QC source identity, separate from frozen physical-case identity."""
import hashlib
from pathlib import Path
from ..core import digest


def reference_execution_identity():
    root=Path(__file__).resolve().parents[3]
    files=('src/pdeep/unified/reference.py','src/pdeep/unified/dataset.py','src/pdeep/unified/provenance.py','src/pdeep/unified/streams.py',
           'src/pdeep/learning_diagnostics/reference_cache.py')
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})
