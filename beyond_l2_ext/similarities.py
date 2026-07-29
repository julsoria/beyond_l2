r"""Isotropic Gaussian similarity layers for the "Gaussian ProtoPNet" architecture
described in Section 4.4 ("Isotropic Gaussian Similarity") of the paper.

Both classes model each prototype as an isotropic Gaussian N(mu_j, sigma_j^2 I) and
differ only in how the squared Mahalanobis distance is mapped to a similarity score:

- `MahalanobisLogDistance` reuses the legacy ProtoPNet heavy-tailed log transform
  S(d) = log((d^2 + 1) / (d^2 + epsilon)), applied to the Mahalanobis distance instead
  of the raw Euclidean one.
- `MahalanobisLogDensity` returns the Gaussian log-density directly,
  S(d) = -0.5 * d^2 - log(Z).

Both are consumed by `explain.extended_explainers.IsotropicLogDistanceFormalExplanation`
and `IsotropicGaussianFormalExplanation` respectively, which map their similarity scores
back into a universal Euclidean space to reuse the base Euclidean Triangle
Inequality / Hypersphere Intersection Approximation machinery (see the paper's
"Isotropic Gaussian Similarity" section).

Not upstreamed into the public `cabrnet` package (as of this writing) — vendored here
as CaBRNet-style plugin `SimilarityLayer` subclasses instead of forking `cabrnet` itself.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from cabrnet.core.utils.similarities import LogDistance, SimilarityLayer


class MahalanobisLogDistance(LogDistance):
    r"""Computes the squared (isotropic/diagonal/general) Mahalanobis distance and maps it
    to a similarity score using the legacy ProtoPNet heavy-tailed logarithmic transformation.
    """

    def __init__(
        self,
        num_prototypes: int,
        num_features: int,
        cov_type: str = "isotropic",
        stability_factor: float = 1e-4,
        **kwargs,
    ) -> None:
        # Inherit from LogDistance to automatically get the distances_to_similarities method
        super().__init__(stability_factor=stability_factor, **kwargs)
        self.cov_type = cov_type
        self.num_features = num_features
        self.num_prototypes = num_prototypes

        # Unconstrained optimization parameters (initialized to identity matrix)
        if self.cov_type == "isotropic":
            self.log_var = nn.Parameter(torch.zeros(num_prototypes))
        elif self.cov_type == "diagonal":
            self.log_var = nn.Parameter(torch.zeros(num_prototypes, num_features))
        elif self.cov_type == "general":
            # Split parameterisation: separate diagonal (log-scale) and off-diagonal
            self.L_diag = nn.Parameter(torch.zeros(num_prototypes, num_features))
            num_offdiag = num_features * (num_features - 1) // 2
            self.L_offdiag = nn.Parameter(torch.zeros(num_prototypes, num_offdiag))
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")

    def reset_parameters(self) -> None:
        r"""Resets the parameters to the identity matrix (log-space 0)."""
        if self.cov_type == "isotropic":
            nn.init.zeros_(self.log_var)
        elif self.cov_type == "diagonal":
            nn.init.zeros_(self.log_var)
        elif self.cov_type == "general":
            nn.init.zeros_(self.L_diag)
            nn.init.zeros_(self.L_offdiag)

    def _get_L_prec(self) -> Tensor:
        r"""Constructs a lower-triangular Cholesky factor using 1D flat indexing
        for unambiguous autograd safety."""
        P, D = self.num_prototypes, self.num_features

        # Diagonal part: exp-parameterised, guaranteed positive
        L_diag = torch.diag_embed(torch.exp(self.L_diag))  # (P, D, D)

        # Off-diagonal part: built via flat 1D index_put, no in-place mutation
        idx = torch.tril_indices(D, D, offset=-1, device=self.L_offdiag.device)
        flat = torch.zeros(P * D * D, device=self.L_offdiag.device, dtype=self.L_offdiag.dtype)
        proto_offsets = (torch.arange(P, device=self.L_offdiag.device) * D * D).unsqueeze(1)  # (P, 1)
        flat_indices = (proto_offsets + idx[0] * D + idx[1]).reshape(-1)  # (P * num_offdiag,)
        flat = flat.index_put((flat_indices,), self.L_offdiag.reshape(-1), accumulate=False)
        L_off = flat.view(P, D, D)

        return L_diag + L_off

    def regularization_loss(self) -> Tensor:
        r"""Computes the KL-Divergence between the learned precision matrices and
        an identity prior: KL(Lambda || I) = 0.5 * (Tr(Lambda) - logdet(Lambda) - D).
        """
        P, D = self.num_prototypes, self.num_features

        if self.cov_type == "isotropic":
            Lambda = torch.exp(-self.log_var)  # (P,)
            trace_Lambda = D * Lambda
            logdet_Lambda = D * (-self.log_var)

        elif self.cov_type == "diagonal":
            Lambda_diag = torch.exp(-self.log_var)  # (P, D)
            trace_Lambda = Lambda_diag.sum(dim=1)
            logdet_Lambda = (-self.log_var).sum(dim=1)  # log(exp(-log_var)) = -log_var

        elif self.cov_type == "general":
            L_prec = self._get_L_prec()  # (P, D, D)
            trace_Lambda = (L_prec**2).sum(dim=(1, 2))  # (P,)
            logdet_Lambda = 2.0 * self.L_diag.sum(dim=1)  # (P,)

        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        kl_div = 0.5 * (trace_Lambda - logdet_Lambda - D)
        return kl_div.mean()

    @property
    def covariances(self) -> Tensor:
        r"""Exposes the true covariance matrices Sigma to the formal explainer."""
        if self.cov_type in ["isotropic", "diagonal"]:
            return torch.exp(self.log_var)
        elif self.cov_type == "general":
            L_prec = self._get_L_prec()
            prec = torch.bmm(L_prec, L_prec.transpose(1, 2))
            return torch.linalg.inv(prec)

    @property
    def precisions(self) -> Tensor:
        r"""Exposes the true precision matrices (Lambda) to the formal explainer.
        Returns a dense tensor of shape (P, D, D) regardless of cov_type."""
        P, D = self.num_prototypes, self.num_features
        device = self.log_var.device if hasattr(self, "log_var") else self.L_diag.device

        if self.cov_type == "isotropic":
            Lambda = torch.exp(-self.log_var)  # (P,)
            return Lambda.view(P, 1, 1) * torch.eye(D, device=device).unsqueeze(0)

        elif self.cov_type == "diagonal":
            Lambda = torch.exp(-self.log_var)  # (P, D)
            return torch.diag_embed(Lambda)

        elif self.cov_type == "general":
            L_prec = self._get_L_prec()
            return torch.bmm(L_prec, L_prec.transpose(1, 2))

    @torch.compiler.disable
    def distances(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise squared Mahalanobis distances (d^2) without algebraic
        expansion, utilizing chunking to prevent VRAM out-of-memory crashes.
        """
        N, D, H, W = features.shape
        P = self.num_prototypes

        z = features.view(N, D, -1).permute(0, 2, 1).unsqueeze(2)  # (N, HW, 1, D)
        mu = prototypes.view(P, D).view(1, 1, P, D)

        chunk_size = 500
        dist_chunks = []

        if self.cov_type in ["isotropic", "diagonal"]:
            Lambda = torch.exp(-self.log_var)
            if self.cov_type == "isotropic":
                Lambda = Lambda.unsqueeze(1).expand(P, D)
            L = Lambda.view(1, 1, P, D)

            for i in range(0, P, chunk_size):
                mu_chunk = mu[:, :, i : i + chunk_size, :]
                L_chunk = L[:, :, i : i + chunk_size, :]
                dist_chunks.append((L_chunk * (z - mu_chunk) ** 2).sum(dim=-1))

        elif self.cov_type == "general":
            L_prec = self._get_L_prec()  # (P, D, D) — Cholesky factor, used directly

            for i in range(0, P, chunk_size):
                mu_chunk = mu[:, :, i : i + chunk_size, :]
                L_chunk = L_prec[i : i + chunk_size]  # (chunk, D, D)

                diff_chunk = z - mu_chunk  # (N, HW, chunk, D)

                # Transform into Cholesky space: ||L^T (z-mu)||^2 — guaranteed non-negative
                transformed = torch.einsum("nxpd, pdc -> nxpc", diff_chunk, L_chunk)
                dist_chunks.append((transformed**2).sum(dim=-1))

        dist = torch.cat(dist_chunks, dim=2)  # (N, HW, P)
        return dist.permute(0, 2, 1).view(N, P, H, W)


class MahalanobisLogDensity(SimilarityLayer):
    r"""Probabilistic (isotropic/diagonal/general) Mahalanobis similarity that outputs
    the Gaussian log-density directly: S(z) = -0.5 * d^2 - log(Z).
    """

    def __init__(self, num_prototypes: int, num_features: int, cov_type: str = "isotropic", **kwargs):
        super().__init__(**kwargs)
        self.cov_type = cov_type
        self.num_features = num_features
        self.num_prototypes = num_prototypes

        if self.cov_type == "isotropic":
            self.log_var = nn.Parameter(torch.zeros(num_prototypes))
        elif self.cov_type == "diagonal":
            self.log_var = nn.Parameter(torch.zeros(num_prototypes, num_features))
        elif self.cov_type == "general":
            self.L_params = nn.Parameter(torch.zeros(num_prototypes, num_features, num_features))
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")

    def _get_L_prec(self) -> Tensor:
        L_tril = torch.tril(self.L_params, diagonal=-1)
        diag_elements = torch.exp(torch.diagonal(self.L_params, dim1=-2, dim2=-1))
        L_diag = torch.diag_embed(diag_elements)
        return L_tril + L_diag

    @property
    def covariances(self) -> Tensor:
        if self.cov_type in ["isotropic", "diagonal"]:
            return torch.exp(self.log_var)
        elif self.cov_type == "general":
            L_prec = self._get_L_prec()
            prec = torch.bmm(L_prec, L_prec.transpose(1, 2))
            return torch.linalg.inv(prec)

    def distances(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        N, D, H, W = features.shape
        P = self.num_prototypes
        mu = prototypes.view(P, D)

        if self.cov_type in ["isotropic", "diagonal"]:
            Lambda = torch.exp(-self.log_var)
            if self.cov_type == "isotropic":
                Lambda = Lambda.unsqueeze(1).expand(P, D)

            term1 = torch.conv2d(features**2, Lambda.view(P, D, 1, 1))
            weight2 = (Lambda * mu).view(P, D, 1, 1)
            term2 = -2.0 * torch.conv2d(features, weight2)
            term3 = (Lambda * mu**2).sum(dim=1).view(1, P, 1, 1)
            return term1 + term2 + term3

        elif self.cov_type == "general":
            L_prec = self._get_L_prec()
            Lambda = torch.bmm(L_prec, L_prec.transpose(1, 2))

            term1 = torch.einsum("n c h w, p c d, n d h w -> n p h w", features, Lambda, features)
            Lambda_mu = torch.einsum("p c d, p d -> p c", Lambda, mu)
            term2 = -2.0 * torch.conv2d(features, Lambda_mu.view(P, D, 1, 1))
            term3 = torch.einsum("p c, p c d, p d -> p", mu, Lambda, mu).view(1, P, 1, 1)
            return term1 + term2 + term3

    def distances_to_similarities(self, distances: Tensor, **kwargs) -> Tensor:
        r"""Maps d^2 directly to the Gaussian log-density."""
        D = self.num_features

        if self.cov_type == "isotropic":
            log_det = D * self.log_var
        elif self.cov_type == "diagonal":
            log_det = self.log_var.sum(dim=1)
        elif self.cov_type == "general":
            L_prec = self._get_L_prec()
            log_det = -2.0 * torch.diagonal(L_prec, dim1=-2, dim2=-1).log().sum(dim=-1)

        log_Z = 0.5 * (D * np.log(2 * np.pi) + log_det)

        # Protect against float precision dropping below 0
        distances = torch.relu(distances)

        # Dynamically adapt shape for (N, P, H, W) or (N, P)
        view_shape = [1, -1] + [1] * (distances.dim() - 2)

        return -0.5 * distances - log_Z.view(*view_shape)
