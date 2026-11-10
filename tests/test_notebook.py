from pathlib import Path
import json
import ast
import re
from pdeep.diagnostics import source_identity


ROOT = Path(__file__).resolve().parents[1]


def test_technical_notebook_sections_and_safe_prefix():
    paths = list((ROOT / "notebooks").glob("*.ipynb"))
    assert {p.name for p in paths} == {"pde_inverse_research.ipynb","01_explore.ipynb","02_train.ipynb","03_evaluate.ipynb"}
    nb = json.loads(paths[0].read_text())
    headings = [re.match(r"## (\d{2})\.", "".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert [m.group(1) for m in headings if m] == [f"{i:02d}" for i in range(1, 16)]
    first_future = next((i for i, c in enumerate(nb["cells"]) if "future-section" in c.get("metadata", {}).get("tags", [])), len(nb["cells"]))
    assert sum(c["cell_type"] == "code" for c in nb["cells"][:first_future]) == 38
    assert nb["metadata"]["pdeep"]["implemented_sections"] == [f"{i:02d}" for i in range(1,16)]
    sources="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
    assert 'TINY_OVERFIT = False' in sources and 'ATrainingConfig(updates=3)' in sources
    assert 'RUN_A_SMOKE = False' in sources and 'C_TINY_OVERFIT=False' in sources
    assert 'CTrainingConfig(updates=3,microbatch=1,effective_batch=3)' in sources
    assert 'RUN_C_SMOKE=False' in sources and 'B_TINY_OVERFIT=False' in sources
    assert 'B_APPROACHES=("B0","B1")' in sources
    assert 'RUN_LEGACY_B_SMOKE=False' in sources
    assert 'FULL_TRAINING_ENABLED=False' in sources and 'LAUNCH_FULL_TRAINING=False' in sources
    assert 'RUN_SHARED_TRAINING_PREFLIGHT=False' in sources
    assert 'RUN_PERSISTENCE_PREFLIGHT=False' in sources
    assert 'NotebookLedger' in sources and 'NOTEBOOK_LEDGER.cell' in sources
    assert 'SUITE_ARCHITECTURES=("A","C","B0","B1","D")' in sources
    assert 'TRAIN_APPROACHES=("A","C","B0","B1")' in sources
    assert not any(c["cell_type"] == "code" for c in nb["cells"][first_future:])
    assert all("NOT IMPLEMENTED" in "".join(c["source"]) for c in nb["cells"][first_future:])


def test_notebook_source_identity_ignores_execution_outputs(tmp_path):
    directory = tmp_path / "notebooks"
    directory.mkdir()
    path = directory / "one.ipynb"
    nb = {"nbformat":4,"cells":[{"cell_type":"code","source":["x=1"],"outputs":[],"execution_count":None,"metadata":{"tags":["implemented"]}}]}
    path.write_text(json.dumps(nb))
    identity = source_identity(tmp_path)["sha256"]
    nb["cells"][0]["outputs"] = [{"output_type":"stream","text":"report"}]
    nb["cells"][0]["execution_count"] = 1
    path.write_text(json.dumps(nb))
    assert source_identity(tmp_path)["sha256"] == identity
    nb["cells"][0]["source"] = ["x=2"]
    path.write_text(json.dumps(nb))
    assert source_identity(tmp_path)["sha256"] != identity


def test_routine_notebooks_safe_short_and_independent():
    for name,expected in (("01_explore.ipynb",13),("02_train.ipynb",15),("03_evaluate.ipynb",13)):
        nb=json.loads((ROOT/"notebooks"/name).read_text())
        assert len(nb["cells"])==expected
        sources="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
        assert "PROJECT_DIR" in sources and "initialize(" in sources
        assert "notebooks/pde_inverse_research" not in sources
        assert all("implemented" in c.get("metadata",{}).get("tags",[]) for c in nb["cells"] if c["cell_type"]=="code")
        if name=="02_train.ipynb":
            assert 'RUN_MODE = "preview"' in sources
            modes=[node.value for node in ast.parse(sources).body if isinstance(node,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='RUN_MODE' for t in node.targets)]
            assert len(modes)==1 and isinstance(modes[0],ast.Constant) and modes[0].value=='preview'
            # Model selection is user configuration; the preview safety guard is mandatory.
            configured=[node.value for node in ast.parse(sources).body if isinstance(node,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='MODELS' for t in node.targets)]
            assert len(configured)==1 and isinstance(configured[0],ast.Tuple) and configured[0].elts
            assert {node.value for node in configured[0].elts}<={'A','B0','B1','C'}
            assert "FULL_TRAINING_ENABLED" not in sources and "LAUNCH_FULL_TRAINING" not in sources
            assert "launch_training(PLAN" in sources and "assert_plan_settings" in sources
        if name=="03_evaluate.ipynb":
            # Preserve explicit user-selected trained exports; no requirement to
            # erase their selection back to the original empty example mapping.
            assignments=[node for node in ast.parse(sources).body if isinstance(node,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id=='CHECKPOINTS' for t in node.targets)]
            assert len(assignments)==1 and isinstance(assignments[0].value,ast.Dict)
            assert "ALLOW_PARTIAL = False" in sources
