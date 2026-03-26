"""
SIGReg (Spectral Information Gain Regulariser) loss module.

Ported and adapted from:
    Quillent, A., Bourdoukan, R. & Löwe, S. ST-JEPA: Provable and Scalable
    Self-Supervised Learning Without the Heuristics. arXiv:2502.01048 (2025).
    https://github.com/LoreCase073/stjepa (2025).

The loss tests whether the marginal distribution of (randomly projected)
representations matches a standard Gaussian by comparing the empirical and
theoretical characteristic functions (CFs) via a weighted L2 distance on a
grid of quadrature knots.  Minimising SIGReg together with an invariance loss
prevents representational collapse while avoiding heuristics such as EMA
stop-gradients, asymmetric architectures, or VICReg-style covariance penalties.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    SIGReg loss module.

    Compares the empirical characteristic function (CF) of randomly
    projected representations to the CF of a standard Gaussian
    N(0, 1) evaluated at a grid of quadrature knots.

    Parameters
    ----------
    knots:
        Number of quadrature knots used for numerical integration over
        the CF domain [0, 3].  The original paper uses 17.
    n_projections:
        Number of random projection directions sampled per forward
        call.  More projections give a lower-variance gradient estimate
        at the cost of memory; 256 matches the original paper.
    """

    def __init__(self, knots: int = 17, n_projections: int = 256):
        super().__init__()
        self.n_projections = n_projections

        # Quadrature knots on [0, 3]
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # Trapezoidal weights * Gaussian window exp(-t^2/2)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)            # [knots]
        self.register_buffer("phi", window)     # CF of N(0,1) at knots
        self.register_buffer("weights", weights * window)  # [knots]

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute the SIGReg loss.

        Parameters
        ----------
        proj:
            Stacked projected representations with shape (V, B, proj_dim).
            V is the number of views, B is the batch size, and proj_dim
            is the output dimension of the projection head.  The tensor
            should be float32 or bfloat16 and live on the correct device
            (CPU or any CUDA device).

        Returns
        -------
        loss:
            Scalar SIGReg loss (non-negative, zero when proj ~ N(0, 1)).
        """
        # -- Random projection onto unit sphere --
        # A: [proj_dim, n_projections]
        A = torch.randn(
            proj.size(-1), self.n_projections,
            device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        # -- Evaluate characteristic function at knots --
        # (proj @ A)           : [V, B, n_projections]
        # .unsqueeze(-1)       : [V, B, n_projections, 1]
        # * self.t             : [V, B, n_projections, knots]
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Empirical CF (average over batch dimension B, i.e. dim -3):
        # real part  → [V, n_projections, knots]
        # imaginary  → [V, n_projections, knots]
        cf_real = x_t.cos().mean(-3)  # mean over B
        cf_imag = x_t.sin().mean(-3)

        # Squared L2 distance between empirical CF and Gaussian CF
        # self.phi: [knots]  →  broadcast over (V, n_projections)
        err = (cf_real - self.phi).square() + cf_imag.square()
        # err: [V, n_projections, knots]

        # Integrate via trapezoidal quadrature and scale by batch size
        # (err @ self.weights): [V, n_projections]
        statistic = (err @ self.weights) * proj.size(-2)  # scale by B

        # Average over views and random directions → scalar
        return statistic.mean()
