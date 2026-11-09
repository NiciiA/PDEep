"""Execute only implemented sections in a fresh kernel, with bounded cell timeouts."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=int, default=60, help="seconds per implemented cell")
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / "notebooks/pde_inverse_research.ipynb"
    notebook = nbformat.read(path, as_version=4)
    prefix = copy.deepcopy(notebook)
    cut = next((i for i, c in enumerate(prefix.cells) if "future-section" in c.metadata.get("tags", [])), len(prefix.cells))
    prefix.cells = prefix.cells[:cut]
    code = [c for c in prefix.cells if c.cell_type == "code"]
    if not code or not all("implemented" in c.metadata.get("tags", []) for c in code):
        raise RuntimeError("prefix contains an unclassified executable cell")
    implemented_sections = notebook.metadata["pdeep"]["implemented_sections"]
    for c in code:
        c.outputs = []
        c.execution_count = None
    start = time.perf_counter()
    # Match this interpreter, including editable installation, instead of a global kernel.
    with tempfile.TemporaryDirectory(prefix="pdeep-kernel-") as tmp:
        kernel_dir = Path(tmp) / "kernels/pdeep-foundations"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "PDEep verification", "language": "python"}))
        env = dict(os.environ)
        env["JUPYTER_PATH"] = tmp + os.pathsep + env.get("JUPYTER_PATH", "")
        env["IPYTHONDIR"] = str(Path(tmp) / "ipython")
        env["MPLCONFIGDIR"] = str(Path(tmp) / "matplotlib")
        manager = AsyncKernelManager(kernel_name="pdeep-foundations",
                                     kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(Path(tmp) / "kernels")]),
                                     connection_file=str(Path(tmp) / "connection.json"))
        client = NotebookClient(prefix, km=manager, timeout=args.timeout, kernel_name="pdeep-foundations",
                                resources={"metadata": {"path": str(root / "notebooks")}}, allow_errors=False)
        try:
            client.execute(env=env, cwd=str(root / "notebooks"))
        finally:
            # Supplying a manager makes notebook-client leave ownership to us.
            import asyncio
            if manager.has_kernel:
                asyncio.run(manager.shutdown_kernel(now=True))
            asyncio.run(manager.cleanup_resources())
    for i, cell in enumerate(prefix.cells):
        notebook.cells[i] = cell
    nbformat.write(notebook, path)
    output = {"status": "passed", "executed_code_cells": len(code), "implemented_sections": implemented_sections,
              "future_sections_executed": 0, "elapsed_seconds": time.perf_counter() - start,
              "interpreter": sys.executable, "outputs": [
                  {"cell_id": c.id, "execution_count": c.execution_count,
                   "outputs": [dict(o) for o in c.outputs]} for c in prefix.cells if c.cell_type == "code"]}
    report = root / "reports/notebook_prefix.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
