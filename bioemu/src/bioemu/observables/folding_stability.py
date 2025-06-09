import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio.PDB import MMCIFParser, PDBParser
from bioemu_benchmarks.eval.folding_free_energies.free_energies import K_BOLTZMANN
from torch._prims_common import DeviceLikeType
from torch_geometric.utils import to_dense_batch

from bioemu.chemgraph import ChemGraph
from bioemu.get_embeds import StrPath, _get_default_embeds_dir, shahexencode


@lru_cache(maxsize=128)
def load_reference_ca_coords(
    ref_path: str,
    device: DeviceLikeType | None = None,
) -> torch.Tensor:
    """
    Load C-alpha coordinates from a reference PDB file into a torch.Tensor.

    This function assumes **nm** as the default unit.

    Args:
        ref_path: Path to the reference PDB file.
        device: torch device for the returned tensor.

    Returns:
        ref_coords: Tensor of shape (L, 3)
    """

    if ref_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif ref_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)  # type: ignore
    else:
        raise ValueError("Unsupported file format. Please provide a .cif or .pdb file.")
    structure = parser.get_structure("ref", ref_path)
    model = next(structure.get_models())
    coords_list = []
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                ca = residue["CA"]
                coords_list.append(ca.get_coord() / 10)
    coords = np.stack(coords_list, axis=0)  # (L, 3)
    return torch.from_numpy(coords).to(device=device, dtype=torch.float32)


def compute_folded_proportion(
    coords: torch.Tensor,
    ref_coords: torch.Tensor,
    k: float = -24.0,
    d_0: float = 0.4,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Compute the expected folded proportion of an ensemble using the f_dRMSD sigmoid metric.

    This function assumes **nm** as the default unit, meaning `d_0` corresponds to 4 Angstroms (0.4 nm).

    Args:
        coords: Tensor of shape (B, L, 3), sampled C-alpha coordinates.
        ref_coords: Tensor of shape (L, 3), reference folded C-alpha coordinates.
        k: Sigmoid steepness parameter.
        d_0: dRMSD threshold (same units as coords, e.g., nm).
        tol: Small value to guard against division by zero.

    Returns:
        p_folded: Tensor of shape (B,), probability of being folded.
    """

    dist_samples = torch.cdist(coords, coords)  # （B, L, L）
    dist_ref = torch.cdist(ref_coords.unsqueeze(0), ref_coords.unsqueeze(0))  # (1, L, L)

    delta = dist_samples - dist_ref
    drmsd = torch.mean(delta**2, dim=(-1, -2)).sqrt()  # (B,)
    p_folded = F.sigmoid(k * (drmsd - d_0))
    p_folded = torch.clamp(p_folded, min=tol, max=1.0 - tol)  # Guard against boundary values

    return p_folded


def compute_dG(
    p_folded: torch.Tensor, temperature: float = 298.0, tol: float = 1e-7
) -> torch.Tensor:
    """
    Compute folding free energy from the folded proportion using Boltzmann relation.

    Args:
        p_folded: Tensor of shape (B,), probability of being folded.
        temperature: Temperature in Kelvin.
        tol: Small value to guard against division by zero.

    Returns:
        delta_G: Tensor of shape (B,), folding free energy in kcal/mol.
    """

    # Guard against boundary values
    p_folded = torch.clamp(p_folded, min=tol, max=1.0 - tol)

    return -K_BOLTZMANN * temperature * torch.logit(p_folded)


def compute_folded_proportion_from_dG(dG: torch.Tensor, temperature: float = 298.0) -> torch.Tensor:
    """
    Compute folded proportion from folding free energy.

    Args:
        dG: Tensor of shape (B,), folding free energy in kcal/mol.
        temperature: Temperature in Kelvin.
    Returns:
        p_folded: Tensor of shape (B,), probability of being folded.
    """

    p_folded = F.sigmoid(-dG / (K_BOLTZMANN * temperature))

    return p_folded


class FoldingStability:
    """Class to compute folding stability metrics from a batch of ChemGraph samples."""

    _dataset_cache: dict[str, pd.DataFrame] = {}  # class-level cache for datasets

    def __init__(
        self,
        dataset_path: StrPath,
        sequence_col: str,
        dG_col: str,
        temperature: float = 298.0,
        k: float = -24.0,
        d_0: float = 0.4,
        tol: float = 1e-7,
        cache_embeds_dir: StrPath | None = None,
    ):
        """Initialize the FoldingStability class.
        Args:
            dataset_path: Path to the dataset (for loading reference free energies).
            sequence_col: Column name in the dataset containing amino acid sequences.
            dG_col: Column name in the dataset containing folding free energies.
            temperature: Temperature in Kelvin for free energy calculations.
            k: Sigmoid steepness parameter for folded proportion.
            d_0: dRMSD threshold (same units as coords, e.g., nm).
            tol: Small value to guard against division by zero.
            cache_embeds_dir: Directory to cache reference PDB files.
        """
        self.k = k
        self.d_0 = d_0
        self.tol = tol
        self.temperature = temperature

        dataset_path = Path(dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path

        # Use cached dataset if available, otherwise load and cache it
        dataset_key = str(dataset_path)
        if dataset_key not in self._dataset_cache:
            self._dataset_cache[dataset_key] = pd.read_csv(dataset_path)
        self.dataset = self._dataset_cache[dataset_key]

        if sequence_col not in self.dataset.columns:
            raise ValueError(f"Column '{sequence_col}' not found in dataset.")
        if dG_col not in self.dataset.columns:
            raise ValueError(f"Column '{dG_col}' not found in dataset.")
        self.sequence_col = sequence_col
        self.dG_col = dG_col

        if cache_embeds_dir is None:
            cache_embeds_dir = _get_default_embeds_dir()
        self.cache_embeds_dir = Path(cache_embeds_dir).expanduser().resolve()
        if not self.cache_embeds_dir.exists():
            raise FileNotFoundError(f"Cache directory {self.cache_embeds_dir} does not exist.")

    def sequence_to_ref_path(self, sequence: str) -> str:
        """
        Convert a sequence to the path of its reference PDB file.

        Uses the same caching pattern as the embeddings system to locate
        the PDB file generated by ColabFold.

        Args:
            sequence: Amino acid sequence.

        Returns:
            Path to the reference PDB file for this sequence.
        """

        seqsha = shahexencode(sequence)
        ref_path = os.path.join(self.cache_embeds_dir, f"{seqsha}.pdb")

        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Reference PDB file not found at {ref_path}. "
                f"Make sure to run sampling first to generate the reference structure."
            )

        return ref_path

    def __call__(self, batch: ChemGraph, sequence: str) -> torch.Tensor:
        """
        Compute the folding stability metric for a batch of ChemGraph samples.

        Args:
            batch: ChemGraph batch containing multiple samples with 'pos' coordinates (B, L, 3).
            sequence: Amino acid sequence corresponding to the batch.

        Returns:
            Tensor of shape (B, 1) containing the ratio.
        """

        ref_path = self.sequence_to_ref_path(sequence)
        coords, _ = to_dense_batch(batch.pos, batch.batch)  # (B, L, 3)
        ref_coords = load_reference_ca_coords(ref_path, device=coords.device)

        # Compute the folded proportion
        p_folded = compute_folded_proportion(
            coords, ref_coords, k=self.k, d_0=self.d_0, tol=self.tol
        )  # (B,)

        return p_folded.unsqueeze(1)  # Return as (B, 1) tensor for consistency
