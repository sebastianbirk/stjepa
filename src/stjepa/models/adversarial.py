"""
Adversarial batch-effect removal via gradient reversal.

Implements the gradient reversal layer from:
    Ganin, Y. et al.  Domain-Adversarial Training of Neural Networks.
    JMLR 17(59):1–35 (2016).

Usage in ST-JEPA
----------------
For each adversarial label (e.g. ``batch_value``, ``assay_value``,
``gene_panel_value``), an ``AdversarialHead`` predicts the label from
the mean-pooled gene-token representation.  A ``GradientReversal`` layer
sits between the encoder output and the head: during the backward pass
it *reverses and scales* the gradient so that the encoder learns to
produce representations from which the label **cannot** be predicted.

The adversarial cross-entropy loss is added to the ST-JEPA loss with a
configurable weight (``adv_weight``).  Setting ``adv_weight=0`` disables
the adversarial loss entirely — no heads are created and no extra
computation is performed.

Lambda scheduling
-----------------
Following Ganin et al., the gradient reversal scaling factor ``lambda``
can be ramped from 0 to 1 over training via a sigmoid schedule::

    lambda(p) = 2 / (1 + exp(-gamma * p)) - 1

where ``p`` goes from 0 to 1 over the total number of training steps
and ``gamma`` controls the steepness (default 10).  This avoids
destabilising the encoder in early training when the classifier is still
random.  Set ``adv_schedule_gamma=0`` to use a fixed lambda of 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient reversal primitive
# ---------------------------------------------------------------------------

class _GradientReversalFn(Function):
    """Autograd function that reverses and scales gradients."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        #print(f"GRL backward called, lambd={ctx.lambd}, grad_norm={grad_output.norm():.4f}") # DEBUG
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    """
    Module wrapper around :class:`_GradientReversalFn`.

    In the forward pass, the input is passed through unchanged.
    In the backward pass, the gradient is reversed and scaled by
    ``lambd``.
    """

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambd)

    def set_lambd(self, lambd: float):
        self.lambd = lambd


# ---------------------------------------------------------------------------
# Adversarial classification head
# ---------------------------------------------------------------------------

class AdversarialHead(nn.Module):
    """
    Small MLP classifier with gradient reversal for a single adversarial
    label.

    Parameters
    ----------
    in_dim:
        Input dimension (encoder embedding size, after pooling).
    n_classes:
        Number of distinct classes for this label.
    hidden_dim:
        Hidden layer width.  Defaults to ``in_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.grad_rev = GradientReversal(lambd=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Pooled encoder representation, shape ``(B, in_dim)``.

        Returns
        -------
        logits:
            Class logits, shape ``(B, n_classes)``.
        """
        return self.classifier(self.grad_rev(x))

    def set_lambd(self, lambd: float):
        """Update the gradient reversal scaling factor."""
        self.grad_rev.set_lambd(lambd)


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------

def adversarial_lambda_schedule(
    step: int,
    total_steps: int,
    gamma: float = 10.0,
) -> float:
    """
    Sigmoid ramp for gradient reversal lambda (Ganin et al., 2016).

    Parameters
    ----------
    step:
        Current training step (0-indexed).
    total_steps:
        Total number of training steps.
    gamma:
        Steepness of the sigmoid.  ``0`` → constant lambda of 1.

    Returns
    -------
    lambd:
        Value in [0, 1].
    """
    if gamma == 0.0 or total_steps <= 1:
        return 1.0
    p = step / max(total_steps - 1, 1)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
