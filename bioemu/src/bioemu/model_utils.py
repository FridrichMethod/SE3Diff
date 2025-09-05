# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path

import hydra
import torch
import yaml

from bioemu.models import DiGConditionalScoreModel
from bioemu.sde_lib import SDE


def load_model(ckpt_path: str | Path, model_config_path: str | Path) -> DiGConditionalScoreModel:
    """Load score model from checkpoint and config."""
    assert os.path.isfile(ckpt_path), f"Checkpoint {ckpt_path} not found"
    assert os.path.isfile(model_config_path), f"Model config {model_config_path} not found"

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    return score_model


def load_sdes(
    model_config_path: str | Path, cache_so3_dir: str | Path | None = None
) -> dict[str, SDE]:
    """Instantiate SDEs from config."""
    with open(model_config_path) as f:
        sdes_config = yaml.safe_load(f)["sdes"]

    if cache_so3_dir is not None:
        sdes_config["node_orientations"]["cache_dir"] = cache_so3_dir

    sdes: dict[str, SDE] = hydra.utils.instantiate(sdes_config)
    return sdes
