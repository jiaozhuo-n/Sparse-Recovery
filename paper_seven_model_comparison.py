"""Seven-method Gaussian sparse recovery: training, SNR tests and paper figures.

The training protocol follows the supplied reference implementation:
  * column-normalized Gaussian sensing matrix;
  * analytical B pretrained with the supplied reference code's rowwise
    top-s coherence surrogate and diagonal normalization;
  * full unfolded model training for 100 epochs;
  * Adam, batch size 64, learning rate 3e-4;
  * validation-based best checkpoint selection;
  * testing at SNR 20, 30, 40 and 50 dB after checkpoint selection;
  * repeated seeds, CSV/JSON artifacts, and three paper comparison figures.

The ALePOM layer and MCP proximal map follow the original paper. The B
pretraining follows the supplied reference implementation; it is a surrogate
for the original paper's ordered-l1 BCD solver. The mAPG candidate recurrence
is Algorithm 2, but its fixed-objective theorem does not directly apply to
this learned network with B that need not equal A.T.

By default, ISTA and FISTA use the same 16-step depth as ALePOM, with a fixed
step and threshold. The optional
validation_tuned protocol reports a separate high-budget comparison. The two
protocols must be reported separately in a paper.
The three 2025
networks use their published update structures, with layer count and dimensions
adapted to this experiment. See REFERENCES and config.json for exact choices.

Windows CMD:
  python paper_eight_model_comparison.py --quick
  python paper_eight_model_comparison.py --seeds 2025
  python paper_eight_model_comparison.py --seeds 2025 2026 2027
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


LEARNED_METHODS = ("ALePOM", "mAPG-ALePOM", "Tail-LISTA", "DU-PSISTA", "IHT-Net")
CLASSICAL_METHODS = ("ISTA", "FISTA")
METHODS = LEARNED_METHODS + CLASSICAL_METHODS
REFERENCES = {
    "ALePOM": {"paper": "Learning Proximal Operator Methods for Nonconvex Sparse Recovery with Theoretical Guarantee"},
    "mAPG-ALePOM": {"paper": "ALePOM combined with mAPG Algorithm 2 from Convergence Analysis of Proximal Gradient with Momentum for Nonconvex"},
    "ISTA": {"paper": "Daubechies, Defrise and De Mol, An iterative thresholding algorithm for linear inverse problems with a sparsity constraint", "year": 2004, "doi": "10.1002/cpa.20042"},
    "FISTA": {"paper": "Beck and Teboulle, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", "year": 2009, "doi": "10.1137/080716542"},
    "Tail-LISTA": {"paper": "Kvich et al., Deep Unfolding of Tail-Based Methods for Robust Sparse Recovery Under Noise and Model Mismatch", "online_year": 2025, "issue_year": 2026, "doi": "10.1109/TNNLS.2025.3638635"},
    "DU-PSISTA": {"paper": "Tokumura et al., Deep Unfolding of Periodically Sketched ISTA", "year": 2025, "doi": "10.1109/APSIPAASC65261.2025.11249150", "extended_derivation": "https://arxiv.org/abs/2604.20214"},
    "IHT-Net": {"paper": "Mohammad-Taheri et al., Soft Greedy Sparse Recovery (IHT-Net)", "year": 2025, "url": "https://arxiv.org/abs/2505.15661v1"},
}
EPS = 1e-12
B_OPTIMIZER = "Reference implementation: Adam on rowwise top-s squared coherence plus Frobenius penalty, with diagonal normalization"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
       5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
       17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
       25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
       29: 2.045, 30: 2.042}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_sensing_matrix(m: int, n: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(m, n, generator=generator) / math.sqrt(m)
    matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(EPS)
    return matrix.to(device)


def sample_amplitudes(
    count: int, n: int, distribution: str, generator: torch.Generator
) -> torch.Tensor:
    if distribution == "gaussian":
        return torch.randn(count, n, generator=generator)
    if distribution == "rademacher":
        values = torch.randint(0, 2, (count, n), generator=generator).float()
        return 2.0 * values - 1.0
    if distribution == "laplace":
        uniform = torch.rand(count, n, generator=generator).clamp(1e-6, 1.0 - 1e-6)
        centered = uniform - 0.5
        scale = 1.0 / math.sqrt(2.0)
        return -scale * torch.sign(centered) * torch.log1p(-2.0 * centered.abs())
    raise ValueError(f"Unknown amplitude distribution: {distribution}")


def add_snr_noise(
    clean_measurement: torch.Tensor,
    snr_db: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if math.isinf(snr_db):
        return clean_measurement
    noise = torch.randn(clean_measurement.shape, generator=generator).to(device)
    signal_norm = clean_measurement.norm(dim=1, keepdim=True).clamp_min(EPS)
    noise_norm = noise.norm(dim=1, keepdim=True).clamp_min(EPS)
    desired_noise_norm = signal_norm / (10.0 ** (snr_db / 20.0))
    return clean_measurement + noise * desired_noise_norm / noise_norm


def make_dataset(
    A: torch.Tensor,
    count: int,
    nonzero_prob: float,
    seed: int,
    device: torch.device,
    amplitude: str = "gaussian",
    snr_db: float = math.inf,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = A.shape[1]
    support = torch.rand(count, n, generator=generator) < nonzero_prob
    values = sample_amplitudes(count, n, amplitude, generator)
    target = values * support
    empty = support.sum(dim=1) == 0
    if empty.any():
        rows = torch.where(empty)[0]
        columns = torch.randint(0, n, (len(rows),), generator=generator)
        target[rows, columns] = 1.0
    target = target.to(device)
    clean_measurement = target @ A.T
    measurement = add_snr_noise(clean_measurement, snr_db, generator, device)
    return target, measurement


def normalize_B(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    diagonal = torch.sum(B * A.T, dim=1, keepdim=True)
    signs = torch.where(diagonal >= 0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
    diagonal = torch.where(diagonal.abs() < 1e-8, signs * 1e-8, diagonal)
    return B / diagonal


def pretrain_B(
    A: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
) -> torch.Tensor:
    path = output_dir / f"B_m{args.m}_seed{seed}.pt"
    if args.reuse_b and path.exists():
        checkpoint = torch.load(path, map_location=A.device, weights_only=True)
        if checkpoint.get("optimizer") not in (None, "reference_coherence"):
            raise ValueError(f"Saved B was made by a different optimizer: {path}")
        if checkpoint.get("m") != args.m or checkpoint.get("n") != args.n or checkpoint.get("seed") != seed:
            raise ValueError(f"Saved B has different matrix dimensions or seed: {path}")
        if checkpoint.get("optimizer") == "reference_coherence" and (
            checkpoint.get("b_lr") != args.b_lr
            or checkpoint.get("b_epochs") != args.b_epochs
            or checkpoint.get("top_s") != args.b_top_s
            or checkpoint.get("fro_weight") != args.b_fro_weight
        ):
            raise ValueError(f"Saved B has different pretraining settings: {path}")
        B = checkpoint["B"].to(A.device)
        if tuple(B.shape) != (args.n, args.m):
            raise ValueError(f"Saved B shape {tuple(B.shape)} does not match {(args.n, args.m)}")
        print(f"[B] loaded {path}", flush=True)
        return normalize_B(B, A).detach()

    initial_B = A.T / A.square().sum(dim=0, keepdim=True).T.clamp_min(EPS)
    parameter = nn.Parameter(initial_B.detach().clone())
    optimizer = torch.optim.Adam([parameter], lr=args.b_lr)
    identity = torch.eye(args.n, device=A.device, dtype=A.dtype)
    top_s = min(max(args.b_top_s, 1), args.n - 1)
    best_loss = float("inf")
    best_B: torch.Tensor | None = None

    for epoch in range(args.b_epochs):
        optimizer.zero_grad(set_to_none=True)
        B = normalize_B(parameter, A)
        G = B @ A - identity
        G = G - torch.diag_embed(torch.diagonal(G))
        largest = torch.topk(G.abs(), k=top_s, dim=1, largest=True, sorted=False).values
        coherence_loss = largest.square().sum(dim=1).mean()
        frobenius_loss = G.square().mean()
        loss = coherence_loss + args.b_fro_weight * frobenius_loss
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_B = normalize_B(parameter.detach(), A).clone()
        interval = max(args.b_epochs // 10, 1)
        if (epoch + 1) % interval == 0 or epoch + 1 == args.b_epochs:
            print(
                f"[B m={args.m} seed={seed}] epoch={epoch + 1:05d}/{args.b_epochs:05d} "
                f"loss={value:.6e}",
                flush=True,
            )

    if best_B is None:
        raise RuntimeError("B pretraining did not produce a checkpoint")
    torch.save(
        {"B": best_B.cpu(), "best_loss": best_loss, "m": args.m, "n": args.n,
         "seed": seed, "optimizer": "reference_coherence", "b_lr": args.b_lr,
         "b_epochs": args.b_epochs, "top_s": args.b_top_s,
         "fro_weight": args.b_fro_weight},
        path,
    )
    return best_B.detach()


def mcp_parameters(raw_theta: torch.Tensor, raw_q: torch.Tensor, raw_gamma: torch.Tensor):
    theta = F.softplus(raw_theta) + 1e-8
    gamma = F.softplus(raw_gamma) + 1e-8
    q = torch.sigmoid(raw_q) * (1.0 - 1e-4)
    eta = q / (2.0 * theta)
    return theta, q, eta, gamma


def mcp_prox(u: torch.Tensor, theta: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Match the supplied ALePOM implementation's per-sample broadcasting.
    theta = theta.expand(u.shape[0], 1) if theta.ndim == 0 else theta.reshape(-1, 1)
    q = (q.expand(u.shape[0], 1) if q.ndim == 0 else q.reshape(-1, 1)).clamp(min=1e-5, max=1.0 - 1e-5)
    absolute = u.abs()
    upper = theta / q
    middle_value = u.sign() * (absolute - theta) / (1.0 - q).clamp_min(1e-6)
    return torch.where(absolute <= theta, torch.zeros_like(u),
                       torch.where(absolute < upper, middle_value, u))


def mcp_penalty(x: torch.Tensor, theta: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    theta = theta.reshape(1, 1)
    q = q.reshape(1, 1).clamp(min=1e-5, max=1.0 - 1e-5)
    eta = q / (2.0 * theta).clamp_min(1e-8)
    upper = 1.0 / (2.0 * eta).clamp_min(1e-8)
    low = x.abs() - eta * x.square()
    high = torch.ones_like(x) / (4.0 * eta).clamp_min(1e-8)
    return torch.where(x.abs() <= upper, low, high).sum(dim=1)


def layer_objective(
    x: torch.Tensor,
    measurement: torch.Tensor,
    A: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    residual = x @ A.T - measurement
    data_term = 0.5 * residual.square().sum(dim=1)
    regularization = (theta / gamma) * mcp_penalty(x, theta, q)
    return data_term + regularization


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).square().sum(dim=1)
    energy = target.square().sum(dim=1).clamp_min(EPS)
    return (error / energy).mean()


def ci95_half_width(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    std = float(values.std(ddof=1))
    return T95.get(len(values) - 1, 1.96) * std / math.sqrt(len(values))


class RecoveryModel(nn.Module):
    """ALePOM Eq. (18), optionally composed with mAPG Algorithm 2."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, args: argparse.Namespace, method: str):
        super().__init__()
        if method not in ("ALePOM", "mAPG-ALePOM"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.layers = args.layers
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())
        self.raw_theta = nn.Parameter(
            torch.full((args.layers,), inverse_softplus(args.theta_init), dtype=A.dtype, device=A.device)
        )
        self.raw_q = nn.Parameter(
            torch.full((args.layers,), inverse_sigmoid(args.q_init), dtype=A.dtype, device=A.device)
        )
        self.raw_gamma = nn.Parameter(
            torch.full((args.layers,), inverse_softplus(args.gamma_init), dtype=A.dtype, device=A.device)
        )
    def layer_parameters(self, layer: int):
        return mcp_parameters(self.raw_theta[layer], self.raw_q[layer], self.raw_gamma[layer])

    def proximal_map(
        self,
        iterate: torch.Tensor,
        measurement: torch.Tensor,
        theta: torch.Tensor,
        q: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        residual = iterate @ self.A.T - measurement
        gamma = gamma.expand(iterate.shape[0], 1)
        theta = theta.expand(iterate.shape[0], 1)
        q = q.expand(iterate.shape[0], 1)
        proximal_input = iterate - gamma * (residual @ self.B.T)
        return mcp_prox(proximal_input, theta, q)

    def forward(
        self,
        measurement: torch.Tensor,
        return_all: bool = False,
        return_auxiliary: bool = False,
    ):
        batch_size = measurement.shape[0]
        iterate = torch.zeros(batch_size, self.A.shape[1], device=measurement.device, dtype=measurement.dtype)
        previous_iterate = torch.zeros_like(iterate)
        previous_accelerated = torch.zeros_like(iterate)
        outputs = []
        acceptance_history = []
        beta_history = []
        t = 1.0
        previous_t = 0.0

        for layer in range(self.layers):
            theta, q, eta, gamma = self.layer_parameters(layer)
            if self.method == "ALePOM":
                next_iterate = self.proximal_map(iterate, measurement, theta, q, gamma)
                accepted = torch.zeros(batch_size, device=measurement.device, dtype=measurement.dtype)
                beta = 0.0
            else:
                beta = (previous_t - 1.0) / t
                extrapolated_point = (
                    iterate
                    + (previous_t / t) * (previous_accelerated - iterate)
                    + beta * (iterate - previous_iterate)
                )
                accelerated = self.proximal_map(extrapolated_point, measurement, theta, q, gamma)
                ordinary = self.proximal_map(iterate, measurement, theta, q, gamma)
                accelerated_value = layer_objective(accelerated, measurement, self.A, theta, q, gamma)
                ordinary_value = layer_objective(ordinary, measurement, self.A, theta, q, gamma)
                choose_accelerated = (accelerated_value <= ordinary_value).detach()
                next_iterate = torch.where(choose_accelerated[:, None], accelerated, ordinary)
                distinct_candidates = (accelerated - ordinary).abs().amax(dim=1) > 1e-12
                accepted = (choose_accelerated & distinct_candidates).to(measurement.dtype)
                previous_accelerated = accelerated

            outputs.append(next_iterate)
            acceptance_history.append(accepted)
            reported_beta = 0.0 if layer == 0 else beta
            beta_history.append(torch.full((batch_size,), reported_beta, device=measurement.device, dtype=measurement.dtype))
            previous_iterate = iterate
            iterate = next_iterate
            previous_t, t = t, (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0

        if not return_all and not return_auxiliary:
            return iterate
        output_stack = torch.stack(outputs, dim=1)
        acceptance_stack = torch.stack(acceptance_history, dim=1)
        beta_stack = torch.stack(beta_history, dim=1)
        if return_all and return_auxiliary:
            return iterate, output_stack, acceptance_stack, beta_stack
        if return_all:
            return iterate, output_stack
        return iterate, acceptance_stack, beta_stack


def soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return values.sign() * F.relu(values.abs() - threshold)


def spectral_lipschitz(A: torch.Tensor) -> float:
    vector = torch.ones(A.shape[1], device=A.device, dtype=A.dtype)
    vector = vector / vector.norm()
    for _ in range(40):
        vector = A.T @ (A @ vector)
        vector = vector / vector.norm().clamp_min(EPS)
    return float((A @ vector).square().sum().clamp_min(EPS))


def classical_step(A: torch.Tensor) -> float:
    """Reference classical baseline step: 0.99 / ||A||_2^2."""
    spectral = float(torch.linalg.matrix_norm(A, ord=2).detach())
    return 0.99 / max(spectral * spectral, EPS)


class TailLISTA(nn.Module):
    """Tail support is updated between LISTA blocks; no oracle support is used."""

    def __init__(self, A: torch.Tensor, args: argparse.Namespace):
        super().__init__()
        self.method = "Tail-LISTA"
        self.layers = args.tail_steps
        self.inner_layers = args.tail_inner_layers
        self.register_buffer("A", A.detach().clone())
        step = 1.0 / spectral_lipschitz(A)
        w1 = step * A.T
        w2 = torch.eye(A.shape[1], device=A.device) - step * (A.T @ A)
        self.W1 = nn.Parameter(w1.unsqueeze(0).repeat(self.inner_layers, 1, 1))
        self.W2 = nn.Parameter(w2.unsqueeze(0).repeat(self.inner_layers, 1, 1))
        self.raw_threshold = nn.Parameter(torch.full((self.inner_layers,), inverse_softplus(args.lista_threshold), device=A.device))

    def forward(self, measurement: torch.Tensor, return_all: bool = False):
        x = measurement.new_zeros((len(measurement), self.A.shape[1]))
        support = torch.zeros_like(x, dtype=torch.bool)
        outputs = []
        for outer in range(self.layers):
            for inner in range(self.inner_layers):
                threshold = F.softplus(self.raw_threshold[inner]) * (~support).to(x.dtype)
                x = soft_threshold(F.linear(x, self.W2[inner]) + F.linear(measurement, self.W1[inner]), threshold)
            outputs.append(x)
            if outer + 1 < self.layers:
                support = torch.zeros_like(support)
                indices = x.detach().abs().topk(outer + 1, dim=1).indices
                support.scatter_(1, indices, True)
        return (x, torch.stack(outputs, dim=1)) if return_all else x


class DUPSISTA(nn.Module):
    """Periodic exact-gradient steps and fixed Gaussian-sketch steps."""

    def __init__(self, A: torch.Tensor, args: argparse.Namespace, seed: int):
        super().__init__()
        self.method = "DU-PSISTA"
        self.layers = args.layers
        self.period = args.sketch_period
        self.register_buffer("A", A.detach().clone())
        generator = torch.Generator(device="cpu").manual_seed(seed + 400000)
        sketch = torch.randn(args.sketch_rows, A.shape[0], generator=generator).to(A.device) / math.sqrt(args.sketch_rows)
        self.register_buffer("S", sketch)
        self.register_buffer("SA", sketch @ A)
        full_step = 1.0 / spectral_lipschitz(A)
        sketch_step = 1.0 / spectral_lipschitz(self.SA)
        initial = [full_step if layer % self.period == 0 else sketch_step for layer in range(args.layers)]
        self.raw_step = nn.Parameter(torch.tensor([inverse_softplus(x) for x in initial], device=A.device))
        self.raw_threshold = nn.Parameter(torch.full((args.layers,), inverse_softplus(args.psista_threshold), device=A.device))

    def forward(self, measurement: torch.Tensor, return_all: bool = False):
        x = measurement.new_zeros((len(measurement), self.A.shape[1]))
        sketched_measurement = measurement @ self.S.T
        outputs = []
        for layer in range(self.layers):
            if layer % self.period == 0:
                gradient = (x @ self.A.T - measurement) @ self.A
            else:
                gradient = (x @ self.SA.T - sketched_measurement) @ self.SA
            step = F.softplus(self.raw_step[layer])
            threshold = F.softplus(self.raw_threshold[layer])
            x = soft_threshold(x - step * gradient, threshold)
            outputs.append(x)
        return (x, torch.stack(outputs, dim=1)) if return_all else x


class IHTNet(nn.Module):
    """Soft-IHT: weighted SoftSort top-k mask on each gradient iterate."""

    def __init__(self, A: torch.Tensor, args: argparse.Namespace):
        super().__init__()
        self.method = "IHT-Net"
        self.layers = args.iht_layers
        self.k = min(args.iht_k, A.shape[1])
        self.tau = args.iht_temperature
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("step", torch.tensor(1.0 / spectral_lipschitz(A), device=A.device))
        self.raw_weight = nn.Parameter(torch.full((A.shape[1],), inverse_softplus(1.0), device=A.device))

    def forward(self, measurement: torch.Tensor, return_all: bool = False):
        x = measurement.new_zeros((len(measurement), self.A.shape[1]))
        outputs = []
        weight = F.softplus(self.raw_weight)
        for layer in range(self.layers):
            u = x + self.step * ((measurement - x @ self.A.T) @ self.A)
            scores = weight * u.abs()
            sorted_top = torch.sort(scores, dim=1, descending=True).values[:, :self.k]
            soft_rows = torch.softmax(-(sorted_top.unsqueeze(2) - scores.unsqueeze(1)).abs() / self.tau, dim=2)
            mask = soft_rows.sum(dim=1)
            x = mask * u
            outputs.append(x)
        return (x, torch.stack(outputs, dim=1)) if return_all else x


def make_learned_model(method: str, A: torch.Tensor, B: torch.Tensor, args: argparse.Namespace, seed: int) -> nn.Module:
    if method in ("ALePOM", "mAPG-ALePOM"):
        return RecoveryModel(A, B, args, method)
    if method == "Tail-LISTA":
        return TailLISTA(A, args)
    if method == "DU-PSISTA":
        return DUPSISTA(A, args, seed)
    if method == "IHT-Net":
        return IHTNet(A, args)
    raise ValueError(method)


@torch.no_grad()
def validation_nmse(
    model: nn.Module,
    target: torch.Tensor,
    measurement: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(target), args.batch_size):
        end = min(start + args.batch_size, len(target))
        prediction = model(measurement[start:end])
        value = normalized_mse(prediction, target[start:end])
        total += float(value) * (end - start)
        count += end - start
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    train_target: torch.Tensor,
    train_measurement: torch.Tensor,
    validation_target: torch.Tensor,
    validation_measurement: torch.Tensor,
    args: argparse.Namespace,
    training_seed: int,
) -> list[dict[str, Any]]:
    if model.method == "IHT-Net":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.iht_lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_validation = validation_nmse(model, validation_target, validation_measurement, args)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(training_seed + epoch)
        permutation = torch.randperm(len(train_target), generator=generator)
        total_final = 0.0
        total_layer = 0.0
        total_objective = 0.0
        total_count = 0

        for start in range(0, len(train_target), args.batch_size):
            indices = permutation[start:start + args.batch_size].to(train_target.device)
            target = train_target.index_select(0, indices)
            measurement = train_measurement.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction, layer_outputs = model(measurement, return_all=True)
            final_loss = normalized_mse(prediction, target)
            layer_loss = torch.stack(
                [normalized_mse(layer_outputs[:, layer, :], target) for layer in range(model.layers)]
            ).mean()
            objective = layer_loss if model.method == "Tail-LISTA" else final_loss + args.layer_loss_weight * layer_loss
            if not torch.isfinite(objective):
                raise RuntimeError("Non-finite training objective; reduce --lr")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = len(target)
            total_final += float(final_loss.detach()) * batch_size
            total_layer += float(layer_loss.detach()) * batch_size
            total_objective += float(objective.detach()) * batch_size
            total_count += batch_size

        validation = validation_nmse(model, validation_target, validation_measurement, args)
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        record = {
            "epoch": epoch + 1,
            "train_final_nmse": total_final / max(total_count, 1),
            "train_layer_nmse": total_layer / max(total_count, 1),
            "train_objective": total_objective / max(total_count, 1),
            "validation_nmse": validation,
            "best_validation_nmse": best_validation,
            "best_epoch": best_epoch,
        }
        history.append(record)
        print(
            f"[{model.method:12s}] epoch={epoch + 1:03d}/{args.epochs:03d} "
            f"train={record['train_final_nmse']:.6e} val={validation:.6e} "
            f"best={best_validation:.6e} (epoch {best_epoch})",
            flush=True,
        )

    model.load_state_dict(best_state)
    print(
        f"[{model.method}] completed {args.epochs} epochs; restored epoch {best_epoch} "
        f"(val={best_validation:.6e})",
        flush=True,
    )
    return history


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    error_energy = (prediction - target).square().sum(dim=1)
    signal_energy = target.square().sum(dim=1).clamp_min(EPS)
    sample_nmse = error_energy / signal_energy
    total_nmse = float(error_energy.sum() / signal_energy.sum().clamp_min(EPS))
    ser_db = 10.0 * math.log10(float(signal_energy.sum() / error_energy.sum().clamp_min(1e-20)))
    true_support = target.abs() > args.support_threshold
    predicted_support = prediction.abs() > args.support_threshold
    tp = (true_support & predicted_support).sum(dim=1).float()
    fp = ((~true_support) & predicted_support).sum(dim=1).float()
    fn = (true_support & (~predicted_support)).sum(dim=1).float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    exact_support = (true_support == predicted_support).all(dim=1).float()
    success = (sample_nmse <= args.success_nmse).float()
    return {
        "ser_db": ser_db,
        "nmse": total_nmse,
        "nmse_db": 10.0 * math.log10(max(total_nmse, 1e-20)),
        "support_f1": float(f1.mean()),
        "exact_support_rate": float(exact_support.mean()),
        "success_rate": float(success.mean()),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    target: torch.Tensor,
    measurement: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, list[float]], torch.Tensor, float]:
    model.eval()
    predictions = []
    layer_predictions: list[list[torch.Tensor]] = [[] for _ in range(model.layers)]
    layer_error = torch.zeros(model.layers, dtype=torch.float64)
    signal_energy = 0.0
    acceptance_sum = torch.zeros(model.layers, dtype=torch.float64)
    beta_sum = torch.zeros(model.layers, dtype=torch.float64)
    total_samples = 0
    inference_seconds = 0.0
    model(measurement[:min(16, len(measurement))])
    if measurement.device.type == "cuda":
        torch.cuda.synchronize()

    for start in range(0, len(target), args.batch_size):
        end = min(start + args.batch_size, len(target))
        batch_target = target[start:end]
        batch_measurement = measurement[start:end]
        if measurement.device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        if model.method in ("ALePOM", "mAPG-ALePOM"):
            prediction, outputs, acceptances, betas = model(
                batch_measurement, return_all=True, return_auxiliary=True
            )
        else:
            prediction, outputs = model(batch_measurement, return_all=True)
            acceptances = outputs.new_zeros((len(batch_measurement), model.layers))
            betas = acceptances
        if measurement.device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - tick
        predictions.append(prediction)
        signal_energy += float(batch_target.square().sum())
        for layer in range(model.layers):
            layer_predictions[layer].append(outputs[:, layer, :])
            layer_error[layer] += float((outputs[:, layer, :] - batch_target).square().sum())
        acceptance_sum += acceptances.sum(dim=0).cpu().double()
        beta_sum += betas.sum(dim=0).cpu().double()
        total_samples += len(batch_target)

    prediction = torch.cat(predictions, dim=0)
    final_metrics = compute_metrics(prediction, target, args)
    layer_rows = []
    for layer in range(model.layers):
        error = float(layer_error[layer])
        layer_prediction = torch.cat(layer_predictions[layer], dim=0)
        layer_metrics = compute_metrics(layer_prediction, target, args)
        layer_rows.append(
            {
                "layer": layer + 1,
                "proximal_calls": 2 * (layer + 1) if model.method == "mAPG-ALePOM" else ((layer + 1) * model.inner_layers if model.method == "Tail-LISTA" else (0 if model.method == "IHT-Net" else layer + 1)),
                "ser_db": 10.0 * math.log10(signal_energy / max(error, 1e-20)),
                "nmse": error / max(signal_energy, EPS),
                "support_f1": layer_metrics["support_f1"],
                "exact_support_rate": layer_metrics["exact_support_rate"],
                "success_rate": layer_metrics["success_rate"],
                "acceptance_rate": float(acceptance_sum[layer] / max(total_samples, 1)),
                "beta": float(beta_sum[layer] / max(total_samples, 1)),
            }
        )
    diagnostics = {
        "acceptance_rate": [row["acceptance_rate"] for row in layer_rows],
        "beta": [row["beta"] for row in layer_rows],
    }
    sample_nmse = (
        (prediction - target).square().sum(dim=1)
        / target.square().sum(dim=1).clamp_min(EPS)
    )
    return final_metrics, layer_rows, diagnostics, sample_nmse.cpu(), inference_seconds


@torch.no_grad()
def classical_path(
    method: str, A: torch.Tensor, measurement: torch.Tensor, checkpoints: tuple[int, ...],
    step: float = 0.0, regularization: float | torch.Tensor = 0.0,
) -> dict[int, torch.Tensor]:
    requested = set(checkpoints)
    x = measurement.new_zeros((len(measurement), A.shape[1]))
    results = {}
    threshold = torch.as_tensor(regularization, device=A.device, dtype=x.dtype) * step
    if threshold.ndim == 1:
        threshold = threshold[:, None]
    extrapolated = x
    t = 1.0
    for iteration in range(1, max(checkpoints) + 1):
        gradient = (extrapolated @ A.T - measurement) @ A
        next_x = soft_threshold(extrapolated - step * gradient, threshold)
        if method == "FISTA":
            next_t = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            extrapolated = next_x + ((t - 1.0) / next_t) * (next_x - x)
            t = next_t
        else:
            extrapolated = next_x
        x = next_x
        if iteration in requested:
            results[iteration] = x
    return results


@torch.no_grad()
def select_classical(
    method: str, A: torch.Tensor, target: torch.Tensor, measurement: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Choose one of two explicitly labeled classical comparison budgets."""
    if args.classical_protocol == "equal_depth":
        return {
            "iterations": int(args.layers),
            "lambda": 0.10,
            "step": classical_step(A),
            "validation_nmse": "",
            "selection_protocol": "equal_depth_fixed_threshold",
        }

    checkpoints = tuple(sorted(set(args.classic_iterations)))
    lambdas = tuple(args.classic_lambda_grid)
    step = classical_step(A)
    scores = {(iteration, lam): [0.0, 0] for iteration in checkpoints for lam in lambdas}
    for start in range(0, len(target), args.batch_size):
        end = min(start + args.batch_size, len(target))
        batch = measurement[start:end]
        stacked = batch.repeat((len(lambdas), 1))
        thresholds = torch.as_tensor(lambdas, device=A.device, dtype=A.dtype).repeat_interleave(len(batch))
        path = classical_path(method, A, stacked, checkpoints, step, thresholds)
        for iteration, prediction in path.items():
            for index, lam in enumerate(lambdas):
                segment = prediction[index * len(batch):(index + 1) * len(batch)]
                value = normalized_mse(segment, target[start:end])
                scores[(iteration, lam)][0] += float(value) * len(batch)
                scores[(iteration, lam)][1] += len(batch)
    selected = min(scores, key=lambda key: scores[key][0] / scores[key][1])
    return {"iterations": selected[0], "lambda": selected[1],
            "step": step, "validation_nmse": scores[selected][0] / scores[selected][1],
            "selection_protocol": "validation_tuned"}


@torch.no_grad()
def evaluate_classical(
    method: str, A: torch.Tensor, settings: dict[str, Any], target: torch.Tensor,
    measurement: torch.Tensor, args: argparse.Namespace,
) -> tuple[dict[str, float], torch.Tensor, float]:
    predictions = []
    elapsed = 0.0
    classical_path(method, A, measurement[:min(16, len(measurement))], (settings["iterations"],),
                   settings["step"], settings["lambda"])
    if A.device.type == "cuda":
        torch.cuda.synchronize()
    for start in range(0, len(target), args.batch_size):
        end = min(start + args.batch_size, len(target))
        if A.device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        result = classical_path(method, A, measurement[start:end], (settings["iterations"],),
                                settings["step"], settings["lambda"])[settings["iterations"]]
        if A.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - tick
        predictions.append(result)
    prediction = torch.cat(predictions)
    sample_nmse = ((prediction - target).square().sum(dim=1) / target.square().sum(dim=1).clamp_min(EPS)).cpu()
    return compute_metrics(prediction, target, args), sample_nmse, elapsed


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    metric_names = (
        "ser_db",
        "nmse",
        "support_f1",
        "exact_support_rate",
        "success_rate",
        "acceptance_rate",
        "beta",
    )
    output = []
    for method in METHODS:
        layers = sorted({int(row["layer"]) for row in rows if row["method"] == method})
        for layer in layers:
            selected = [row for row in rows if row["method"] == method and int(row["layer"]) == layer]
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in selected], dtype=float)
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                half = ci95_half_width(values)
                output.append(
                    {
                        "method": method,
                        "layer": layer,
                        "proximal_calls": selected[0]["proximal_calls"],
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "ci95_low": mean - half,
                        "ci95_high": mean + half,
                        "num_seeds": len(seeds),
                    }
                )
    return output


def aggregate_training(history_rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    metrics = ("train_final_nmse", "train_layer_nmse", "train_objective", "validation_nmse")
    output = []
    for method in METHODS:
        epochs = sorted({int(row["epoch"]) for row in history_rows if row["method"] == method})
        for epoch in epochs:
            selected = [
                row for row in history_rows
                if row["method"] == method and int(row["epoch"]) == epoch
            ]
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in selected], dtype=float)
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                half = ci95_half_width(values)
                output.append(
                    {
                        "method": method,
                        "epoch": epoch,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "ci95_low": mean - half,
                        "ci95_high": mean + half,
                        "num_seeds": len(seeds),
                    }
                )
    return output








def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_seven_models_snr20_50")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--train-nonzero-prob", type=float, default=0.1)
    parser.add_argument("--test-nonzero-prob", type=float, default=0.1)
    parser.add_argument("--snr-db", type=float, default=math.inf)
    parser.add_argument("--amplitude", choices=("gaussian", "rademacher", "laplace"), default="gaussian")
    parser.add_argument("--b-epochs", type=int, default=2000, help="Reference-style B pretraining epochs")
    parser.add_argument("--b-lr", type=float, default=3e-3)
    parser.add_argument("--b-top-s", type=int, default=50)
    parser.add_argument("--b-fro-weight", type=float, default=1e-3)
    parser.add_argument("--reuse-b", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--layer-loss-weight", type=float, default=0.05)
    parser.add_argument("--theta-init", type=float, default=0.10)
    parser.add_argument("--q-init", type=float, default=0.50, help="q=2*theta*eta initialization from the reference code")
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--tail-steps", type=int, default=None, help="Defaults to expected support size n * train-nonzero-prob")
    parser.add_argument("--tail-inner-layers", type=int, default=2)
    parser.add_argument("--lista-threshold", type=float, default=0.05)
    parser.add_argument("--sketch-rows", type=int, default=100)
    parser.add_argument("--sketch-period", type=int, default=4)
    parser.add_argument("--psista-threshold", type=float, default=0.05)
    parser.add_argument("--iht-k", type=int, default=50)
    parser.add_argument("--iht-layers", type=int, default=30)
    parser.add_argument("--iht-lr", type=float, default=1e-2)
    parser.add_argument("--iht-temperature", type=float, default=0.01)
    parser.add_argument("--classical-protocol", choices=("equal_depth", "validation_tuned"),
                        default="equal_depth", help="Budget for ISTA/FISTA; report each protocol separately")
    parser.add_argument("--classic-iterations", type=int, nargs="+", default=[16, 64, 256, 1024, 4096],
                        help="ISTA/FISTA iteration candidates only for validation_tuned protocol")
    parser.add_argument("--classic-lambda-grid", type=float, nargs="+", default=[0.0, 0.003, 0.01, 0.03, 0.1],
                        help="ISTA/FISTA threshold coefficients only for validation_tuned protocol")
    parser.add_argument("--support-threshold", type=float, default=1e-4)
    parser.add_argument("--success-nmse", type=float, default=1e-4)
    parser.add_argument("--snr-list", type=float, nargs="+", default=[20, 30, 40, 50],
                        help="Test SNR values; --snr-db still controls training/validation noise")
    parser.add_argument("--layer-snr", type=float, default=50,
                        help="SNR used for the first-16-layer SER figure; must be in --snr-list")
    parser.add_argument("--error-bars", choices=("std", "sem", "ci95", "none"), default="std")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--input-dir", type=Path, help="Saved experiment for test-only/plots-only modes")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test-only", action="store_true", help="Test saved checkpoints, without training")
    mode.add_argument("--plots-only", action="store_true", help="Redraw three figures from saved CSVs")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")
    if len(args.snr_list) != len(set(args.snr_list)) or any(not math.isfinite(s) or not 20 <= s <= 50 for s in args.snr_list):
        parser.error("Use unique test SNR values in [20, 50]")
    if args.layer_snr not in args.snr_list:
        parser.error("--layer-snr must be included in --snr-list")
    if args.dpi < 1:
        parser.error("dpi must be positive")
    if args.tail_steps is None:
        args.tail_steps = max(1, min(args.n, round(args.n * args.train_nonzero_prob)))
    if args.quick:
        if args.output_dir == Path(__file__).resolve().parent / "results_seven_models_snr20_50":
            args.output_dir = Path(__file__).resolve().parent / "results_seven_models_snr20_50_quick"
        args.n_train, args.n_val, args.n_test = 512, 128, 128
        args.epochs, args.b_epochs = 3, 30
        args.seeds = [args.seeds[0]]
        if args.classical_protocol == "validation_tuned":
            args.classic_iterations = [args.layers]
        args.tail_steps, args.tail_inner_layers = 2, 2
        args.iht_layers = 4
    positive = [args.m, args.n, args.layers, args.n_train, args.n_val, args.n_test, args.batch_size, args.epochs, args.b_epochs]
    if any(value < 1 for value in positive):
        parser.error("All dimensions, counts, and epoch settings must be positive")
    if not 0.0 < args.m / args.n < 1.0:
        parser.error("Require m < n for sparse recovery")
    if not 0.0 < args.train_nonzero_prob < 1.0 or not 0.0 < args.test_nonzero_prob < 1.0:
        parser.error("Sparsity probabilities must be in (0, 1)")
    if not 0.0 < args.q_init < 1.0:
        parser.error("q-init must be in (0, 1)")
    if args.b_lr <= 0.0 or args.b_fro_weight < 0.0 or args.b_top_s < 1:
        parser.error("b-lr and b-top-s must be positive; b-fro-weight cannot be negative")
    if not args.classic_iterations or any(i < 1 for i in args.classic_iterations):
        parser.error("Classical iterations must be positive")
    if any(value < 0 for value in args.classic_lambda_grid) or not args.classic_lambda_grid:
        parser.error("Classical regularization values must be nonnegative")
    if not (1 <= args.tail_steps <= args.n and 1 <= args.iht_k <= args.n and args.iht_layers >= 1 and args.iht_lr > 0):
        parser.error("Tail steps and IHT sparsity level must be in [1, n]")
    if not (1 <= args.sketch_rows <= args.m) or args.tail_inner_layers < 1 or args.sketch_period < 1:
        parser.error("Invalid tail-inner-layers, sketch-period or sketch dimension")
    if min(args.lista_threshold, args.psista_threshold, args.iht_temperature) <= 0:
        parser.error("Learned thresholds and IHT temperature must be positive")
    return args


def run_seed(seed: int, args: argparse.Namespace, device: torch.device, output_dir: Path):
    print("\n" + "=" * 88, flush=True)
    print(f"seed={seed}, M={args.m}, N={args.n}, layers={args.layers}, device={device}", flush=True)
    print("=" * 88, flush=True)
    set_seed(seed)
    A = make_sensing_matrix(args.m, args.n, seed, device)
    b_started = time.perf_counter()
    B = normalize_B(pretrain_B(A, args, output_dir, seed), A).detach()
    b_pretrain_seconds = time.perf_counter() - b_started
    diagonal_error = float((torch.diagonal(B @ A) - 1.0).abs().max())
    with torch.no_grad():
        off_diagonal = B @ A
        off_diagonal = off_diagonal - torch.diag_embed(torch.diagonal(off_diagonal))
        top_s = min(max(args.b_top_s, 1), args.n - 1)
        coherence = float(
            off_diagonal.abs().topk(top_s, dim=1, sorted=False).values.square().sum(dim=1).mean()
        )
    print(
        f"[B audit] max |diag(BA)-1|={diagonal_error:.6e}, "
        f"rowwise top-{top_s} coherence={coherence:.6e}",
        flush=True,
    )

    train_target, train_measurement = make_dataset(
        A, args.n_train, args.train_nonzero_prob, seed + 1, device, args.amplitude, args.snr_db
    )
    validation_target, validation_measurement = make_dataset(
        A, args.n_val, args.train_nonzero_prob, seed + 2, device, args.amplitude, args.snr_db
    )
    test_seed = seed + 100000 + args.m

    set_seed(seed)
    initial_model = RecoveryModel(A, B, args, "ALePOM")
    initial_state = copy.deepcopy(initial_model.state_dict())
    del initial_model
    all_history = []
    models = {}
    training_info = {}
    selections = {}
    for method in LEARNED_METHODS:
        set_seed(seed)
        model = make_learned_model(method, A, B, args, seed)
        if method in ("ALePOM", "mAPG-ALePOM"):
            model.load_state_dict(initial_state)
        tick = time.perf_counter()
        history = train_model(model, train_target, train_measurement,
                              validation_target, validation_measurement, args, seed)
        training_info[method] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": time.perf_counter() - tick, "tuning_seconds": 0.0,
            "b_pretrain_seconds": b_pretrain_seconds if method in ("ALePOM", "mAPG-ALePOM") else 0.0,
            "selected_iterations": args.tail_steps * args.tail_inner_layers if method == "Tail-LISTA" else
                                   (args.iht_layers if method == "IHT-Net" else args.layers),
            "selected_lambda": "",
        }
        all_history.extend({"seed": seed, "method": method, **r} for r in history)
        checkpoint = {"method": method, "seed": seed, "A": A.detach().cpu(), "B": B.detach().cpu(),
                      "state_dict": model.state_dict(), "args": vars(args),
                      "best_epoch": history[-1]["best_epoch"], "test_seed": test_seed}
        torch.save(checkpoint, output_dir / f"seed_{seed:04d}_{method.replace('-', '_')}.pt")
        models[method] = model.eval()
    for method in CLASSICAL_METHODS:
        tick = time.perf_counter()
        selections[method] = select_classical(method, A, validation_target, validation_measurement, args)
        settings = selections[method]
        training_info[method] = {"parameters": 0, "training_seconds": 0.0,
                                 "tuning_seconds": time.perf_counter() - tick,
                                 "b_pretrain_seconds": 0.0,
                                 "selected_iterations": settings["iterations"],
                                 "selected_lambda": settings["lambda"]}
    result = evaluate_snr_conditions(seed, A, models, selections, training_info, args)
    result["history"] = all_history
    result["selections"] = [{"seed": seed, "method": method, **settings}
                            for method, settings in selections.items()]
    result["diagonal_error"] = diagonal_error
    result["coherence"] = coherence
    for name, key in (("metrics", "layers"), ("training", "history"), ("costs", "costs"),
                      ("classical_selection", "selections"), ("paired", "paired"),
                      ("paired_vs_mapg", "paired_vs_mapg"), ("diagnostics", "diagnostics")):
        write_records(output_dir / f"seed_{seed:04d}_{name}.csv", result[key])
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result








PLOT_METHODS = ("ALePOM", "mAPG-ALePOM", "ISTA", "FISTA", "Tail-LISTA", "DU-PSISTA", "IHT-Net")
STYLES = {
    "ALePOM": ("#1f77b4", "o"), "mAPG-ALePOM": ("#d62728", "s"),
    "ISTA": ("#9467bd", "v"), "FISTA": ("#8c564b", "^"),
    "Tail-LISTA": ("#17becf", "D"), "DU-PSISTA": ("#2ca02c", "P"),
    "IHT-Net": ("#e377c2", "<"),
}
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def stats(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite data in plot input")
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    ci = T95.get(len(a) - 1, 1.96) * std / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return float(a.mean()), std, ci

def aggregate_records(rows, keys, metrics):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, group in groups.items():
        if len({int(r["seed"]) for r in group}) != len(group):
            raise ValueError(f"Duplicate seed records for {key}")
        row = dict(zip(keys, key))
        row["num_seeds"] = len(group)
        for m in metrics:
            mean, std, ci = stats([float(r[m]) for r in group])
            row.update({m + "_mean": mean, m + "_std": std, m + "_ci95": ci})
        out.append(row)
    return out

def uncertainty(row, metric, opt):
    """Return the selected visual uncertainty measure.

    The underlying CSV always stores mean, standard deviation, and 95% CI.
    This only changes how uncertainty is drawn in figures.
    """
    if opt.error_bars == "none":
        return 0.0
    if opt.error_bars == "std":
        return float(row[metric + "_std"])
    if opt.error_bars == "sem":
        return float(row[metric + "_std"]) / math.sqrt(max(int(row["num_seeds"]), 1))
    return float(row[metric + "_ci95"])

def style_axes(ax, xlabel, ylabel, title, ratio=False, log=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, framealpha=0.95)
    if ratio:
        ax.set_ylim(-0.02, 1.02)
    if log:
        ax.set_yscale("log")

def plot_paper_curves(rows, family, metric, out, opt, filename_suffix="", title_suffix=""):
    selected = [r for r in rows if r["family"] == family]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for method in PLOT_METHODS:
        data = sorted([r for r in selected if r["method"] == method], key=lambda r: float(r["x"]))
        if not data:
            continue
        x = np.array([float(r["x"]) for r in data])
        mean = np.array([r[metric + "_mean"] for r in data])
        ci = np.array([uncertainty(r, metric, opt) for r in data])
        color, marker = STYLES[method]
        lower = mean - ci
        upper = mean + ci
        if metric == "nmse":
            lower = np.maximum(lower, 1e-20)
            mean = np.maximum(mean, 1e-20)
        if metric in ("support_f1", "exact_support_rate", "success_rate"):
            lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        if family == "layer":
            ax.plot(x, mean, color=color, marker=marker, markersize=4,
                    markevery=max(len(x) // 16, 1), label=method, linewidth=1.6)
            ax.fill_between(x, lower, upper, color=color, alpha=0.08)
        else:
            ax.errorbar(x, mean, yerr=np.array([mean - lower, upper - mean]),
                        color=color, marker=marker, markersize=4, linewidth=1.5,
                        capsize=3, label=method)
    xlabel = {"layer": "Output step (Tail-LISTA: 2 inner updates per step)",
              "sparsity": "Nonzero probability", "snr": "Measurement SNR (dB)",
              "amplitude": "Nonzero amplitude distribution"}[family]
    if family == "amplitude":
        items = sorted({(float(r["x"]), r["condition"]) for r in selected})
        ax.set_xticks([x for x, _ in items], [c.capitalize() for _, c in items])
    elif family in ("snr", "sparsity"):
        ax.set_xticks(sorted({float(r["x"]) for r in selected}))
    style_axes(ax, xlabel, METRICS[metric],
               "",
               ratio=metric in ("support_f1", "exact_support_rate", "success_rate"), log=metric == "nmse")
    save_paper_figure(fig, out, f"{metric}_{family}{filename_suffix}", opt)

def plot_paper_training(history, out, opt):
    """Plot positive training metrics on log axes without misleading bands.

    A continuous confidence band is visually unstable on a log axis when the
    first few epochs are many orders of magnitude larger than the converged
    values. Draw the mean trajectory as a clean line and add small capped
    error bars at sparse checkpoints instead. This keeps the uncertainty
    visible without producing vertical ribbons or filled triangles.
    """
    for metric in ("validation_nmse",):
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        for method in LEARNED_METHODS:
            rows = [r for r in history if r["method"] == method]
            data = aggregate_records(rows, ("method", "epoch"), (metric,))
            data.sort(key=lambda r: int(r["epoch"]))
            if not data:
                continue
            x = np.array([int(r["epoch"]) for r in data])
            mean = np.maximum(np.array([r[metric + "_mean"] for r in data]), 1e-20)
            ci = np.maximum(np.array([uncertainty(r, metric, opt) for r in data]), 0.0)
            color, marker = STYLES[method]
            ax.plot(x, mean, color=color, marker=marker, markevery=max(len(x) // 12, 1),
                    label=method, markersize=3, linewidth=1.6, zorder=3)
            # Sparse CI bars avoid large filled regions caused by log scaling.
            stride = max(len(x) // 8, 1)
            checkpoint = np.arange(0, len(x), stride)
            lower = np.maximum(mean[checkpoint] - ci[checkpoint], 1e-20)
            upper = np.maximum(mean[checkpoint] + ci[checkpoint], lower)
            if opt.error_bars != "none":
                ax.errorbar(x[checkpoint], mean[checkpoint],
                            yerr=np.vstack((mean[checkpoint] - lower, upper - mean[checkpoint])),
                            fmt="none", ecolor=color, elinewidth=0.8, capsize=2,
                            alpha=0.55, zorder=2)
        style_axes(ax, "Training epoch", metric.replace("_", " ").capitalize(),
                   "", log=True)
        save_paper_figure(fig, out, "training_" + metric, opt)




METRICS = {"ser_db": "SER (dB; higher is better)", "nmse": "NMSE (lower is better)",
           "support_f1": "Support F1", "exact_support_rate": "Exact support recovery rate",
           "success_rate": "Recovery success rate"}


def write_records(path, rows):
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(path, rows, fields)


@torch.no_grad()
def evaluate_classical_layers(method, A, settings, target, measurement, args):
    # Store the first 16 outputs and the selected final output, including
    # validation-tuned runs whose iteration budget is larger than 16.
    steps = int(settings["iterations"])
    checkpoints = tuple(sorted(set(range(1, min(max(16, args.layers), steps) + 1)) | {steps}))
    outputs = {k: [] for k in checkpoints}
    elapsed = 0.0
    classical_path(method, A, measurement[:min(16, len(measurement))], (steps,),
                   settings["step"], settings["lambda"])
    for start in range(0, len(target), args.batch_size):
        if A.device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        path = classical_path(method, A, measurement[start:start + args.batch_size], checkpoints,
                              settings["step"], settings["lambda"])
        if A.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - tick
        for k in checkpoints:
            outputs[k].append(path[k])
    layers = []
    for k in checkpoints:
        prediction = torch.cat(outputs[k])
        layers.append({"layer": k, "proximal_calls": k,
                       **compute_metrics(prediction, target, args), "acceptance_rate": 0.0, "beta": 0.0})
    prediction = torch.cat(outputs[steps])
    errors = ((prediction - target).square().sum(dim=1) / target.square().sum(dim=1).clamp_min(EPS)).cpu()
    return compute_metrics(prediction, target, args), layers, {}, errors, elapsed


@torch.no_grad()
def evaluate_snr_conditions(seed, A, models, selections, training_info, args):
    result = {key: [] for key in ("final", "layers", "costs", "diagnostics", "paired", "paired_vs_mapg")}
    test_seed = seed + 100000 + args.m
    for snr in args.snr_list:
        # Resetting the dataset generator gives identical clean targets and
        # identical noise directions at all SNRs. Only noise amplitude changes.
        target, measurement = make_dataset(A, args.n_test, args.test_nonzero_prob,
                                           test_seed, A.device, args.amplitude, snr)
        final_by_method, errors_by_method = {}, {}
        for method in METHODS:
            if method in LEARNED_METHODS:
                final, layers, diagnostics, errors, elapsed = evaluate_model(models[method], target, measurement, args)
            else:
                final, layers, diagnostics, errors, elapsed = evaluate_classical_layers(
                    method, A, selections[method], target, measurement, args)
            if not all(math.isfinite(float(v)) for v in final.values()):
                raise RuntimeError(f"Non-finite test metrics: seed={seed}, method={method}, SNR={snr}")
            final_by_method[method], errors_by_method[method] = final, errors
            result["final"].append({"seed": seed, "method": method, "snr_db": snr,
                                     "family": "snr", "condition": f"snr={snr:g}", "x": snr, **final})
            result["layers"].extend({"seed": seed, "method": method, "snr_db": snr, **r} for r in layers)
            result["diagnostics"].extend({"seed": seed, "method": method, "snr_db": snr,
                                            "layer": r["layer"], "acceptance_rate": r["acceptance_rate"],
                                            "beta": r["beta"]} for r in layers)
            result["costs"].append({"seed": seed, "method": method, "snr_db": snr,
                                     **training_info.get(method, {}), "inference_seconds": elapsed,
                                     "inference_ms_per_sample": 1000.0 * elapsed / len(target)})
            print(f"[test SNR={snr:g}] {method}: SER={final['ser_db']:.4f} dB, NMSE={final['nmse']:.6e}", flush=True)
        proposed_error = errors_by_method["mAPG-ALePOM"]
        for competitor in METHODS:
            if competitor == "mAPG-ALePOM":
                continue
            other_error = errors_by_method[competitor]
            pair = {"seed": seed, "snr_db": snr,
                    "ser_gain_db": final_by_method["mAPG-ALePOM"]["ser_db"] - final_by_method[competitor]["ser_db"],
                    "sample_win_rate": float((proposed_error < other_error).float().mean()),
                    "median_nmse_ratio": float((proposed_error / other_error.clamp_min(EPS)).median())}
            result["paired_vs_mapg"].append({**pair, "competitor": competitor})
            if competitor == "ALePOM":
                result["paired"].append(pair)
    return result


def save_paper_figure(fig, directory, name, options):
    """Original 10.5 x 5.6 layout; no automatic title and original method names."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for ax in fig.axes:
            ax.set_title("")
        fig.tight_layout()
        for fmt in options.formats:
            destination = directory / f"{name}.{fmt}"
            for suffix in ("", "_updated", "_updated_2"):
                candidate = directory / f"{name}{suffix}.{fmt}"
                try:
                    fig.savefig(candidate, dpi=options.dpi, bbox_inches="tight")
                except PermissionError:
                    continue
                if candidate != destination:
                    print(f"File is locked; saved as {candidate}", flush=True)
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    raise RuntimeError(f"Empty figure: {candidate}")
                break
            else:
                raise PermissionError(f"Cannot save figure: {destination}")
    finally:
        plt.close(fig)


def make_paper_figures(output_dir, final_rows, layer_rows, history, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    figures_dir = output_dir / "paper_figures"
    snr_aggregate = aggregate_records(final_rows, ("family", "condition", "x", "method"), tuple(METRICS))
    # Select a single, explicitly recorded SNR for the layer figure; never mix
    # observations from different noise levels as if they were seed replicates.
    selected_layers = [{**r, "family": "layer", "condition": str(r["layer"]), "x": int(r["layer"])}
                       for r in layer_rows if float(r["snr_db"]) == args.layer_snr and int(r["layer"]) <= 16]
    layer_aggregate = aggregate_records(selected_layers, ("family", "condition", "x", "method"), tuple(METRICS))
    write_records(output_dir / "snr_aggregate.csv", snr_aggregate)
    write_records(output_dir / "layer_aggregate.csv", layer_aggregate)
    plot_paper_curves(snr_aggregate, "snr", "ser_db", figures_dir, args)
    plot_paper_curves(layer_aggregate, "layer", "ser_db", figures_dir, args, filename_suffix="_1_16")
    plot_paper_training(history, figures_dir, args)
    expected = ("training_validation_nmse", "ser_db_snr", "ser_db_layer_1_16")
    (figures_dir / "figure_settings.json").write_text(json.dumps({
        "models": list(PLOT_METHODS), "layer_snr_db": args.layer_snr,
        "layer_axis": "Native output steps; Tail-LISTA has multiple inner updates per step",
        "error_bars": args.error_bars, "canvas_inches": [10.5, 5.6],
        "titles": False, "figure_names": list(expected)}, indent=2), encoding="utf-8")


def load_test_models(source_dir, seed, args, device):
    # Local checkpoints from this experiment contain pathlib-valued args.
    first = torch.load(source_dir / f"seed_{seed:04d}_ALePOM.pt", map_location="cpu", weights_only=False)
    A, B = first["A"].to(device), first["B"].to(device)
    if tuple(A.shape) != (args.m, args.n):
        raise ValueError(f"Wrong sensing dimensions for seed={seed}")
    models = {}
    for method in LEARNED_METHODS:
        checkpoint = first if method == "ALePOM" else torch.load(
            source_dir / f"seed_{seed:04d}_{method.replace('-', '_')}.pt", map_location="cpu", weights_only=False)
        if checkpoint["method"] != method or int(checkpoint["seed"]) != seed:
            raise ValueError(f"Wrong checkpoint identity: {method}, seed={seed}")
        if not torch.equal(checkpoint["A"], first["A"]) or not torch.equal(checkpoint["B"], first["B"]):
            raise ValueError(f"Checkpoints do not share the same A/B: {method}, seed={seed}")
        model = make_learned_model(method, A, B, args, seed).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.requires_grad_(False)
        models[method] = model.eval()
    rows = read_csv(source_dir / "classical_selection.csv")
    selections = {}
    for r in rows:
        if int(r["seed"]) == seed and r["method"] in CLASSICAL_METHODS:
            selections[r["method"]] = {"iterations": int(r["iterations"]),
                                         "step": float(r["step"]), "lambda": float(r["lambda"]),
                                         "validation_nmse": r["validation_nmse"],
                                         "selection_protocol": r["selection_protocol"]}
    if set(selections) != set(CLASSICAL_METHODS):
        raise ValueError(f"Missing classical settings for seed={seed}")
    return A, B, models, selections


def main():
    args = parse_args()
    source_dir = (args.input_dir or args.output_dir).resolve()
    output_dir = args.output_dir.resolve()
    saved_config = None
    if args.test_only or args.plots_only:
        saved_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
        # Restore saved model/training settings. Only evaluation and rendering
        # choices may be overridden, so a test never changes the architecture.
        overrides = {k: getattr(args, k) for k in
                     ("snr_list", "layer_snr", "error_bars", "formats", "dpi", "device", "test_only", "plots_only", "input_dir", "output_dir")}
        for key in vars(args):
            if key in saved_config and key not in overrides:
                setattr(args, key, saved_config[key])
        for key, value in overrides.items():
            setattr(args, key, value)
        if args.input_dir is not None and "--output-dir" not in __import__("sys").argv:
            output_dir = source_dir
    if args.plots_only:
        final = read_csv(source_dir / "snr_metrics.csv")
        layers = read_csv(source_dir / "layer_metrics.csv")
        history = read_csv(source_dir / "training_history.csv")
        make_paper_figures(output_dir, final, layers, history, args)
        print(f"Figures written to {output_dir / 'paper_figures'}", flush=True)
        return
    if args.input_dir is not None and not args.test_only:
        raise ValueError("Use --input-dir only with --test-only or --plots-only")
    if not args.test_only and (output_dir / "config.json").exists():
        raise RuntimeError(f"Results already exist in {output_dir}; choose a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    started = time.perf_counter()
    combined = {key: [] for key in ("final", "layers", "history", "costs", "selections", "diagnostics", "paired", "paired_vs_mapg")}
    diagonal_errors, b_coherences = {}, {}
    for seed in args.seeds:
        if args.test_only:
            A, B, models, selections = load_test_models(source_dir, seed, args, device)
            result = evaluate_snr_conditions(seed, A, models, selections, {}, args)
            result["history"] = [r for r in read_csv(source_dir / "training_history.csv")
                                   if int(r["seed"]) == seed and r["method"] in LEARNED_METHODS]
            result["selections"] = [{"seed": seed, "method": method, **setting} for method, setting in selections.items()]
            result["diagonal_error"] = float((torch.diagonal(B @ A) - 1.0).abs().max())
            result["coherence"] = saved_config.get("b_rowwise_top_s_coherence", {}).get(str(seed))
            del models, A, B
        else:
            result = run_seed(seed, args, device, output_dir)
        for key in combined:
            combined[key].extend(result[key])
        diagonal_errors[str(seed)] = result["diagonal_error"]
        b_coherences[str(seed)] = result["coherence"]
    for filename, key in (("snr_metrics", "final"), ("seed_metrics", "layers"), ("layer_metrics", "layers"),
                           ("training_history", "history"), ("costs", "costs"), ("classical_selection", "selections"),
                           ("diagnostics", "diagnostics"), ("paired_results", "paired"), ("paired_vs_mapg", "paired_vs_mapg")):
        write_records(output_dir / f"{filename}.csv", combined[key])
    long_aggregate = []
    for snr in args.snr_list:
        selected = [r for r in combined["layers"] if r["snr_db"] == snr]
        long_aggregate.extend({"snr_db": snr, **r} for r in aggregate_rows(selected, args.seeds))
    write_records(output_dir / "aggregate_metrics.csv", long_aggregate)
    write_records(output_dir / "aggregate_training.csv", aggregate_training(combined["history"], args.seeds)
                  if not args.test_only else aggregate_records(combined["history"], ("method", "epoch"),
                       ("train_final_nmse", "train_layer_nmse", "train_objective", "validation_nmse")))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(EXPERIMENT_NOTES)
    config.update({"script_version": "seven-method integrated SNR experiment v3", "device": str(device),
                   "methods": list(METHODS), "references": REFERENCES, "b_optimizer": B_OPTIMIZER,
                   "diagonal_errors": diagonal_errors, "b_rowwise_top_s_coherence": b_coherences,
                   "data_split": "Independent train/validation/test splits; test never used for fitting or checkpoint selection",
                   "test_sampling": "Same clean targets and noise directions across SNR; same measurements across methods",
                   "layer_plot_snr_db": args.layer_snr,
                   "source_dir": str(source_dir) if args.test_only else None})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    make_paper_figures(output_dir, combined["final"], combined["layers"], combined["history"], args)
    final_aggregate = aggregate_records(combined["final"], ("snr_db", "method"), tuple(METRICS))
    paired_aggregate = aggregate_records(combined["paired"], ("snr_db",),
                                         ("ser_gain_db", "sample_win_rate", "median_nmse_ratio"))
    competitor_aggregate = aggregate_records(combined["paired_vs_mapg"], ("snr_db", "competitor"),
                                             ("ser_gain_db", "sample_win_rate", "median_nmse_ratio"))
    summary = {"output_dir": str(output_dir), "classical_protocol": args.classical_protocol,
               "seeds": args.seeds, "snr_list": args.snr_list, "layer_snr_db": args.layer_snr,
               "final_metrics": final_aggregate, "paired": paired_aggregate,
               "paired_vs_mapg": competitor_aggregate, "elapsed_seconds": time.perf_counter() - started,
               "figures": [f"paper_figures/{name}.{fmt}" for name in
                           ("training_validation_nmse", "ser_db_snr", "ser_db_layer_1_16") for fmt in args.formats]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed. Results and three paper figures: {output_dir}", flush=True)

EXPERIMENT_NOTES = {'training_losses': 'Tail-LISTA: mean outer-step NMSE as Eq. (18)-(19); other learned models: final NMSE plus layer_loss_weight times mean intermediate NMSE, matching the existing ALePOM comparison protocol.', 'classical_reference_step': '0.99 / ||A||_2^2', 'classical_reference_threshold': '0.1 * step', 'adaptations': {'Tail-LISTA': 'Support starts empty and grows by one largest current-magnitude index per outer step, as in the source. To reach expected sparsity 50 on 200x500 data, default outer steps equal round(n * train_nonzero_prob) and each has two LISTA folds (100 total); source used 12 outer steps with 8 or 12 folds on 32x64 data with sparsity 4-12. Each inner fold has learned matrices and threshold shared across outer steps; loss averages outer estimates.', 'DU-PSISTA': 'Fixed Gaussian sketch seeded independently of data, exact gradient every sketch_period layers, otherwise sketched gradient; learned positive per-layer steps and thresholds.', 'IHT-Net': "SoftSort over nonnegative weighted absolute gradient iterates, sum first iht_k soft permutation rows as support mask; shared coordinate weights learned with RMSprop, gradient step fixed at reciprocal spectral norm squared, 30 layers by default. Paper's structured-support data replaced by the common unstructured Gaussian task.", 'classical': "equal_depth: ISTA/FISTA use K=layers, step=0.99/||A||_2^2, threshold=0.1*step. validation_tuned: classical iteration count and regularization selected on validation NMSE. These are different budgets and should be reported separately. Neither setting is claimed to reproduce the original ALePOM paper's undocumented classical tuning exactly."}, 'mcp_parameterization': 'q=2*theta*eta; theta=softplus(raw_theta), q=sigmoid(raw_q), gamma=softplus(raw_gamma)', 'mapg_rule': 'Algorithm 2: y_k=x_k+(t_(k-1)/t_k)(z_k-x_k)+((t_(k-1)-1)/t_k)(x_k-x_(k-1)); compare F(z_(k+1)) and F(v_(k+1))', 'theory_scope': 'Layer-dependent learned parameters and B != A.T mean the fixed-objective mAPG theorem is not asserted for this network'}

if __name__ == "__main__":
    main()
