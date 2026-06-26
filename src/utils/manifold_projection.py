"""Lightweight SDF manifold projection utilities.

The default mode uses an exact forward projection and a cheaper backward pass
based on a first-order approximation. When enabled, the module can also fall
back to a stricter implicit-differentiation style backward that reuses the
cached local differential quantities.

Key components:
- LightweightSDF: 3-layer SIREN MLP for SDF prediction.
- ManifoldProjection: custom autograd Function (forward exact, backward configurable).
- ManifoldProjectionController: high-level wrapper that manages the SDF net,
  the projection mode, and exposes a clean interface for GaussianModel integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

_DELTA = 1e-6


# ---------------------------------------------------------------------------
# SIREN layer and lightweight SDF network
# ---------------------------------------------------------------------------

class SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            bound = 1 / self.in_features if self.is_first else math.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class LightweightSDF(nn.Module):
    """3-layer SIREN SDF network: 3 -> 64 -> 64 -> 1."""

    def __init__(self, hidden_dim: int = 64, omega_0: float = 30.0):
        super().__init__()
        self.net = nn.Sequential(
            SirenLayer(3, hidden_dim, is_first=True, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                bound = math.sqrt(6 / last.in_features) / 30.0
                last.weight.uniform_(-bound, bound)
                last.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(-1, 3)).squeeze(-1)


@dataclass
class _ProjectionCache:
    mu: torch.Tensor
    phi: torch.Tensor
    g: torch.Tensor
    H: torch.Tensor
    h: torch.Tensor


# ---------------------------------------------------------------------------
# ManifoldProjection autograd Function
# ---------------------------------------------------------------------------

class ManifoldProjection(torch.autograd.Function):
    """Project centres onto the SDF zero level-set.

    Backward modes:
    - "approx": first-order approximation (default, cheapest).
    - "implicit": linear solve using cached J (medium cost).
    - "strict": full closed-form J^T (expensive, for reference / small-scale testing only).
    """

    @staticmethod
    def _compute_local_differentials(mu: torch.Tensor, sdf: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mu.dim() != 2 or mu.shape[-1] != 3:
            raise ValueError(f"Expected mu shape [N, 3], got {tuple(mu.shape)}")

        mu_req = mu.detach().requires_grad_(True)
        phi = sdf(mu_req)
        if phi.dim() != 1 or phi.shape[0] != mu.shape[0]:
            raise ValueError(f"SDF must return shape [N], got {tuple(phi.shape)}")

        # If the SDF is not producing a differentiable output (e.g. eval/inference
        # mode or a detached path), fall back to a safe gradient-free projection.
        if not phi.requires_grad:
            g = torch.zeros_like(mu_req)
            h = torch.ones(mu.shape[0], device=mu.device, dtype=mu.dtype)
            proj = mu_req
            return proj, mu_req.detach(), phi.detach(), g, torch.zeros(mu.shape[0], 3, 3, device=mu.device, dtype=mu.dtype), h

        grad_list, hess_list = [], []
        for i in range(mu.shape[0]):
            gi = torch.autograd.grad(phi[i], mu_req, create_graph=True, retain_graph=True)[0][i]
            grad_list.append(gi)
            rows = [torch.autograd.grad(gi[j], mu_req, retain_graph=True)[0][i] for j in range(3)]
            hess_list.append(torch.stack(rows, dim=0))

        g = torch.stack(grad_list, dim=0)
        H = torch.stack(hess_list, dim=0)
        h = g.square().sum(dim=-1) + _DELTA
        proj = mu_req - (phi / h).unsqueeze(-1) * g
        return proj, mu_req.detach(), phi.detach(), g.detach(), H.detach(), h.detach()

    @staticmethod
    def forward(ctx, mu: torch.Tensor, sdf: nn.Module, backward_mode: str = "approx"):
        proj, mu0, phi, g, H, h = ManifoldProjection._compute_local_differentials(mu, sdf)
        ctx.backward_mode = backward_mode
        ctx.save_for_backward(mu0, phi, g, H, h)
        return proj.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mu, phi, g, H, h = ctx.saved_tensors
        if grad_output is None:
            return None, None, None

        v = grad_output
        mode = ctx.backward_mode

        if mode == "strict":
            # Full closed-form Jacobian from the user specification.
            g_col = g.unsqueeze(-1)
            g_row = g.unsqueeze(-2)
            ggT = g_col @ g_row
            term3 = torch.matmul(ggT, v.unsqueeze(-1)).squeeze(-1) / h.unsqueeze(-1)
            term2 = torch.matmul(H.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1) * (phi / h).unsqueeze(-1)
            term4 = torch.matmul((ggT @ H).transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1) \
                     * (2.0 * phi / h.square()).unsqueeze(-1)
            return v - term2 - term3 + term4, None, None

        if mode == "implicit":
            # Build local J (3x3 per point) and solve J^T v = grad.
            a = (phi / h).unsqueeze(-1)
            I = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0).expand(v.shape[0], 3, 3).clone()
            ggT = g.unsqueeze(-1) * g.unsqueeze(-2)
            J = I - a.unsqueeze(-1) * H - ggT / h.unsqueeze(-1).unsqueeze(-1)
            J = J + (2.0 * phi / h.square()).unsqueeze(-1).unsqueeze(-1) * (ggT @ H)
            grad_mu = torch.linalg.solve(J.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)
            return grad_mu, None, None

        # Default: "approx" — first-order approximation without explicit J materialisation.
        # Keeps cached g, h, phi for the projection but avoids a full Hessian contraction.
        # This is the recommended mode for large-scale training.
        g_norm_sq = h - _DELTA
        alpha = (phi / h).unsqueeze(-1)
        gg_v = (g * v).sum(dim=-1, keepdim=True)
        grad_mu = v - alpha * (g * (gg_v / h.unsqueeze(-1)))
        grad_mu = grad_mu - (phi / h).unsqueeze(-1) * torch.einsum("nij,nj->ni", H, v)
        return grad_mu, None, None


def project_mu(mu: torch.Tensor, sdf: nn.Module, backward_mode: str = "approx") -> torch.Tensor:
    return ManifoldProjection.apply(mu, sdf, backward_mode)


# ---------------------------------------------------------------------------
# High-level controller (the recommended integration point for GaussianModel)
# ---------------------------------------------------------------------------

class ManifoldProjectionController:
    """Manages the SDF net, projection mode, and GaussianModel integration.

    Usage:
        controller = ManifoldProjectionController(cfg)  # reads cfg.mapping.manifold_projection.*
        proj_xyz = controller.project(pc.get_xyz)       # returns projected centers
        sdf_losses = controller.sdf_regularization()    # optional SDF regularizer
        controller.training_setup(optimizer_params)     # add SDF params to optimizer
    """

    VALID_MODES = ("approx", "implicit", "strict")

    def __init__(self, cfg: Optional[object] = None):
        # Read config with safe defaults
        if cfg is not None and hasattr(cfg, "mapping"):
            mp_cfg = cfg.mapping.get("manifold_projection", {})
        elif cfg is not None and hasattr(cfg, "manifold_projection"):
            mp_cfg = cfg.manifold_projection
        else:
            mp_cfg = {}

        self.enabled: bool = bool(mp_cfg.get("enabled", False))
        self.backward_mode: str = mp_cfg.get("backward_mode", "approx")
        self.hidden_dim: int = int(mp_cfg.get("hidden_dim", 64))
        self.omega_0: float = float(mp_cfg.get("omega_0", 30.0))
        self.sdf_lr: float = float(mp_cfg.get("sdf_lr", 1e-3))
        self.sdf_reg_weight: float = float(mp_cfg.get("sdf_reg_weight", 1e-4))

        if self.backward_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid backward_mode={self.backward_mode}, must be one of {self.VALID_MODES}")

        self.sdf: Optional[LightweightSDF] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        if self.enabled:
            self.sdf = LightweightSDF(hidden_dim=self.hidden_dim, omega_0=self.omega_0)
            self.sdf.train()

    def project(self, xyz: torch.Tensor) -> torch.Tensor:
        """Project raw Gaussian centres to the SDF manifold.

        If disabled, returns xyz unchanged.
        """
        if not self.enabled or self.sdf is None:
            return xyz
        if next(self.sdf.parameters()).device != xyz.device:
            self.sdf = self.sdf.to(xyz.device)
        return project_mu(xyz, self.sdf, backward_mode=self.backward_mode)

    def sdf_regularization(self, xyz: torch.Tensor) -> torch.Tensor:
        """Eikonal regularizer: encourage ||∇φ(x)|| ≈ 1 away from the surface."""
        if not self.enabled or self.sdf is None:
            return torch.tensor(0.0, device=xyz.device, dtype=xyz.dtype)

        xyz_req = xyz.detach().requires_grad_(True)
        phi = self.sdf(xyz_req)
        grad = torch.autograd.grad(phi.sum(), xyz_req, create_graph=True)[0]
        return ((grad.norm(dim=-1) - 1.0) ** 2).mean() * self.sdf_reg_weight

    def get_normals(self, xyz: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute SDF gradients as surface normals at given points.

        Args:
            xyz: Point coordinates [N, 3]
            normalize: Whether to normalize gradients to unit length

        Returns:
            Surface normals [N, 3], pointing outward from the surface
        """
        if not self.enabled or self.sdf is None:
            return torch.zeros_like(xyz)

        if xyz.numel() == 0:
            return xyz

        xyz_req = xyz.detach().requires_grad_(True)
        sdf_values = self.sdf(xyz_req)
        grad = torch.autograd.grad(
            sdf_values.sum(), xyz_req, create_graph=False, retain_graph=False
        )[0]

        if normalize:
            grad_norm = grad.norm(dim=-1, keepdim=True)
            grad = grad / (grad_norm + 1e-8)

        return grad

    def training_setup(self, opt_params: object) -> None:
        """Register SDF parameters with the existing Gaussian optimizer.

        opt_params should expose .xyzOptimizer and similar attribute names
        matching the structure created by GaussianModel.training_setup().
        """
        if not self.enabled or self.sdf is None or opt_params is None:
            return

        sdf_params = list(self.sdf.parameters())
        if not sdf_params:
            return

        # Attach a dedicated LR for SDF weights (independent of Gaussian params)
        sdf_opt = torch.optim.Adam(sdf_params, lr=self.sdf_lr)
        if hasattr(opt_params, "register_extra_optimizer"):
            opt_params.register_extra_optimizer("sdf", sdf_opt)
        else:
            # Fallback: stash on the controller; caller should extend step manually
            self._sdf_optimizer = sdf_opt

    def step(self) -> None:
        """Perform one SDF optimizer step. Call after GaussianModel optimizer.step()."""
        if hasattr(self, "_sdf_optimizer") and self._sdf_optimizer is not None:
            self._sdf_optimizer.step()
            self._sdf_optimizer.zero_grad()

    @property
    def device(self) -> torch.device:
        return next(self.sdf.parameters()).device if self.sdf is not None else torch.device("cpu")

    def state_dict(self) -> dict:
        return {"sdf": self.sdf.state_dict() if self.sdf else None} if self.enabled else {}

    def load_state_dict(self, state: dict) -> None:
        if self.enabled and "sdf" in state and state["sdf"] is not None and self.sdf is not None:
            self.sdf.load_state_dict(state["sdf"])
