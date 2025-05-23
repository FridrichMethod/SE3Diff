import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType
from tqdm.auto import trange

from bioemu.denoiser import EulerMaruyamaPredictor
from bioemu.so3_sde import (
    SO3SDE,
    DiGSO3SDE,
    angle_from_rotmat,
    apply_rotvec_to_rotmat,
    rotmat_to_rotvec,
)
from se3diff.train import _get_so3_score, igso3_expansion


class EulerMaruyamaPredictorFinetune(EulerMaruyamaPredictor):

    def reverse_drift_and_diffusion_with_controller(
        self,
        u: torch.Tensor,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score_weight = (
            0.5 * self.marginal_concentration_factor * (1 + self.noise_weight**2)
        )
        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        drift = drift - diffusion**2 * score * score_weight
        drift = drift + diffusion * u * score_weight

        return drift, diffusion

    def update_given_drift_and_diffusion_with_noise(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.randn_like(drift)
        dW = self.noise_weight * torch.sqrt(dt.abs()) * z

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        if isinstance(self.corruption, SO3SDE):
            mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
            sample = apply_rotvec_to_rotmat(
                mean,
                diffusion * dW,
                tol=self.corruption.tol,
            )
        else:
            mean = x + drift * dt
            sample = x + diffusion * dW
        return sample, mean, dW

    def update_given_finetune_score(
        self,
        u: torch.Tensor,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion_with_controller(
            u=u, x=x, t=t, batch_idx=batch_idx, score=score
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion_with_noise(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=diffusion,
        )


def reverse_finetune_diffusion(
    sde: DiGSO3SDE,
    score_model: nn.Module,
    finetune_model: nn.Module,
    *,
    device: DeviceLikeType | None = None,
    batch_size: int = 4096,
    num_steps: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    sde.to(device)
    score_model.to(device).eval()
    finetune_model.to(device)  # need not be in eval mode

    x_t = sde.prior_sampling((batch_size, 3, 3), device=device)
    predictor = EulerMaruyamaPredictorFinetune(
        corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
    )
    dt = -1.0 / torch.tensor(num_steps, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    xs_list = [x_t]
    dWs_list = []
    us_list = []

    for i in trange(num_steps, desc="Reverse diffusion", leave=False):
        t_val = t_vals[i].item()
        t = torch.full((batch_size,), t_val, device=device)
        X_t = rotmat_to_rotvec(x_t)

        with torch.no_grad():
            score = _get_so3_score(X_t, sde, score_model, t)  # (B,3)

        # may not with torch.no_grad()
        u_t = finetune_model(X_t, t)  # (B,3)

        x_t, _, dW_t = predictor.update_given_finetune_score(
            u_t, x=x_t, t=t, dt=dt, batch_idx=None, score=score  # type: ignore
        )
        xs_list.append(x_t)
        dWs_list.append(dW_t)
        us_list.append(u_t)

    xs = torch.stack(xs_list, dim=0)  # (T+1,B,3,3)
    dWs = torch.stack(dWs_list, dim=0)  # (T,B,3)
    us = torch.stack(us_list, dim=0)  # (T,B,3)

    return xs, t_vals, us, dWs


def assign_igso3(
    x_0: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    weights: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the the probability of each Gaussian component for each sample in x_0.
    Args:
        x_0: Tensor of shape (B,3,3) representing the rotation matrices.
        mus: Tensor of shape (K,3,3) representing the means of the Gaussian components.
        sigmas: Tensor of shape (K,) representing the standard deviations of the Gaussian components.
        weights: Tensor of shape (K,) representing the weights of the Gaussian components.
    Returns:
        Tensor of shape (B,K) representing the probabilities of each Gaussian component for each sample in x_0.
    """

    x_0_rel = torch.einsum("k...ij,b...il->bk...jl", mus, x_0)  # (B,K,3,3)
    x_0_rel_angle = angle_from_rotmat(x_0_rel)[0]  # (B,K)
    l_grid = torch.arange(l_max, device=x_0.device)
    pdf = igso3_expansion(x_0_rel_angle, sigmas, l_grid, tol=tol) * weights  # (B,K)
    probs = pdf / (torch.sum(pdf, dim=-1, keepdim=True) + tol)  # (B,K)

    return probs


def compute_importance_weights(
    t_vals: torch.Tensor, us: torch.Tensor, dWs: torch.Tensor
) -> torch.Tensor:
    """Computes the importance weights for the reverse diffusion process.
    Args:
        t_vals: Tensor of shape (T+1,) representing the time values.
        us: Tensor of shape (T,B,3) representing the fine-tuned term.
        dWs: Tensor of shape (T,B,3) representing the Wiener increments.
    Returns:
        Tensor of shape (B,) representing the importance weights.
    """

    diff = us - us.detach()  # (T,B,3)
    dts = torch.diff(t_vals)  # (T,)

    # Integrate from t=1 to t=0, i.e., reverse time
    int_diff_dW = torch.einsum("tb...i,tb...i->b...", diff, -dWs)  # (B,)
    int_diff_sq_dt = torch.einsum("tb...i,tb...i,t->b...", diff, diff, -dts)  # (B,)
    ws = torch.exp(int_diff_dW - int_diff_sq_dt / 2)  # (B,)

    return ws


def compute_ev_loss(
    ws: torch.Tensor, probs: torch.Tensor, weights: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:
    """Compute the expected value loss for the fine-tuning process.
    Args:
        ws: Tensor of shape (B,) representing the importance weights.
        probs: Tensor of shape (B, K) representing the probabilities of each Gaussian component for each sample in x_0.
        weights: Tensor of shape (K,) representing the weights of the Gaussian components.
    Returns:
        Tensor representing the expected value loss.
    """

    B = ws.shape[0]
    h_weights = ws.unsqueeze(1) * (probs - weights)  # (B,K)
    # h_weights = ws.unsqueeze(1) * probs - weights  # (B,K)

    pbar = torch.mean(probs.detach(), dim=0)  # detach in case track grad
    # stab = torch.sum(pbar, dim=0) / (pbar + tol)  # (K,)
    stab = torch.ones_like(pbar)
    stab = stab / torch.mean(stab)  # (K,)

    # Compute the importance sample estimator
    # (\sum_{i\neq j} h_i h_j) / (B(B-1))
    loss_weights = (
        (torch.sum(h_weights, dim=0) ** 2 - torch.sum(h_weights**2, dim=0))
        * stab
        / (B * (B - 1))
    )

    # TODO: Include mus and sigmas in the loss
    loss_ev = torch.sum(loss_weights)

    return loss_ev


def compute_kl_loss(
    t_vals: torch.Tensor, us: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence loss for the fine-tuning process.
    Args:
        t_vals: Tensor of shape (T+1,) representing the time values.
        us: Tensor of shape (T,B,3) representing the fine-tuned term.
        ws: Tensor of shape (B,) representing the importance weights.
    Returns:
        Tensor representing the KL divergence loss.
    """
    dts = torch.diff(t_vals)  # (T,)

    # Integrate from t=1 to t=0 i.e., reverse time
    weighted_kl = torch.einsum("b...,tb...i,tb...i,t->b...", ws, us, us, -dts)  # (B,)
    B = weighted_kl.shape[0]
    baseline = (weighted_kl.sum() - weighted_kl) / (B - 1)
    loss_kl = (
        torch.mean(weighted_kl - baseline.detach()) / 2
    )  # TODO: figure out the use of rloo
    # loss_kl = torch.mean(weighted_kl) / 2

    return loss_kl


def compute_finetune_loss(
    sde: DiGSO3SDE,
    score_model: nn.Module,
    finetune_model: nn.Module,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    weights: torch.Tensor,
    device: DeviceLikeType | None = None,
    lambda_: float = 0.1,
    batch_size: int = 4096,
    num_steps: int = 200,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:

    with torch.no_grad():
        xs, t_vals, _, dWs = reverse_finetune_diffusion(
            sde,
            score_model,
            finetune_model,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
        )

    Xs = rotmat_to_rotvec(xs)  # (T+1,B,3)
    us = finetune_model(Xs[:-1], t_vals[:-1].unsqueeze(-1))  # (T+1,B,3)
    # us_list = []
    # for i in trange(num_steps, desc="Reverse diffusion", leave=False):
    #     t_val = t_vals[i].item()
    #     t = torch.full((batch_size,), t_val, device=device)
    #     X_t = Xs[i]  # (B,3)
    #     u_t = finetune_model(X_t, t)  # (B,3)
    #     us_list.append(u_t)
    # us = torch.stack(us_list, dim=0)  # (T,B,3)

    probs = assign_igso3(xs[-1], mus, sigmas, weights, l_max=l_max, tol=tol)  # (B,K)
    ws = compute_importance_weights(t_vals, us, dWs)  # (B,)
    loss_ev = compute_ev_loss(ws, probs, weights, tol=tol)
    loss_kl = compute_kl_loss(t_vals, us, ws)
    loss = loss_ev + lambda_ * loss_kl

    return loss
