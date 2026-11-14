from dataclasses import replace
from pathlib import Path
from PIL import Image
from pdeep.geometric.models import create
from pdeep.unified.plots import reconstruction_examples,dataset_plots


def test_bounded_composite_reconstruction_and_coverage(arithmetic_dataset,tmp_path):
    # ZERO arithmetic targets and UNTRAINED verification model: rendering only.
    arithmetic_dataset.config=replace(arithmetic_dataset.config,stream_grid=4)
    model=create('A_GEO',arithmetic_dataset.norms,'verification',seed=42)
    output=reconstruction_examples(model,arithmetic_dataset,tmp_path/'examples.png')
    assert model.training and Image.open(output).size[1]>1000
    plots=dataset_plots(arithmetic_dataset.config,tmp_path)
    assert len(plots)==2 and all(Path(p).stat().st_size>1000 for p in plots)
