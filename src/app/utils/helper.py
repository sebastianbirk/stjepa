"""
Helper functions for ST-JEPA model initialisation and training on spatial
transcriptomics data.

Adapted from Quillent, A., Bourdoukan, R. & Löwe, S.  ST-JEPA: Provable
and Scalable Self-Supervised Learning Without the Heuristics.
arXiv:2502.01048 (2025).
"""

import logging
import sys
from typing import Literal

import torch
import torch.nn as nn

import stjepa.models.gene_transformers as gt
from stjepa.models.multimask import EncoderMultiMaskWrapper
from stjepa.models.utils import trunc_normal_
from stjepa.utils.schedulers import (CosineWDSchedule,
                                        WarmupCosineSchedule)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ---------------------------------------------------------------------------
# ST-JEPA-specific helpers
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    3-layer MLP projection head used in ST-JEPA.

    Maps encoder pooled representations (shape ``[B, in_dim]``) to a
    lower-dimensional projection space (shape ``[B, proj_dim]``) via two
    hidden layers with optional normalisation and GELU activations.

    Follows the architecture from:
        Quillent, A., Bourdoukan, R. & Löwe, S.  ST-JEPA: Provable and
        Scalable Self-Supervised Learning Without the Heuristics.
        arXiv:2502.01048 (2025).

    Parameters
    ----------
    in_dim:
        Input dimension (encoder embedding size).
    hidden_dim:
        Hidden layer width.  Defaults to 4 × in_dim.
    proj_dim:
        Output projection dimension.
    norm:
        Normalisation layer to apply after each hidden linear layer.
        ``'bn'`` → ``BatchNorm1d`` (convert to SyncBatchNorm for DDP),
        ``'ln'`` → ``LayerNorm``,  ``'none'`` → no normalisation.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        hidden_dim: int | None = None,
        norm: Literal["bn", "ln", "none"] = "bn",
    ):
        super().__init__()
        hidden_dim = hidden_dim or (in_dim * 4)

        def _norm_layer(dim: int) -> nn.Module:
            if norm == "bn":
                return nn.BatchNorm1d(dim)
            elif norm == "ln":
                return nn.LayerNorm(dim)
            else:
                return nn.Identity()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=(norm != "bn")),
            _norm_layer(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=(norm != "bn")),
            _norm_layer(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim, bias=True),
        )

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Pooled encoder representations of shape ``(B, in_dim)``.

        Returns
        -------
        proj:
            Projected representations of shape ``(B, proj_dim)``.
        """
        return self.layers(x)


def init_stjepa_model(
    gt_type: Literal["rank", "counts", "combined"],
    count_encoding: Literal["value_bins", "mlp"],
    n_value_bins: int,
    cell_pos_enc: Literal["segment", "coord"],
    device: str,
    vocab_size: int,
    seq_len: int,
    n_special_tokens: int,
    n_segments: int,
    proj_dim: int,
    n_special_values: int | None = None,
    enc_emb_dim: int = 768,
    enc_depth: int = 12,
    proj_hidden_dim: int | None = None,
    proj_norm: Literal["bn", "ln", "none"] = "bn",
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    use_flash_attention: bool = True,
    use_layer_norm: bool = True,
    sep_gene_tokens_neb: bool = False,
    pos_learnable: bool = False,
    nz_spc: bool = False,
    mlp_bias: bool = False,
) -> tuple[EncoderMultiMaskWrapper, "ProjectionHead"]:
    """
    Initialise the ST-JEPA encoder and projection head.

    ST-JEPA is a single-network, no-EMA method: only an encoder and a
    projection head are created (no predictor, no target encoder).

    Parameters
    ----------
    gt_type, count_encoding, n_value_bins, cell_pos_enc, device,
    vocab_size, seq_len, n_special_tokens, n_segments,
    n_special_values, enc_emb_dim, enc_depth, num_heads, mlp_ratio,
    use_flash_attention, use_layer_norm, sep_gene_tokens_neb,
    pos_learnable, nz_spc, mlp_bias:
        Gene transformer encoder configuration parameters.
    proj_dim:
        Output dimensionality of the projection head.
    proj_hidden_dim:
        Hidden width of the projection head MLP.  Defaults to
        ``4 × enc_emb_dim``.
    proj_norm:
        Normalisation in the projection head: ``'bn'`` (BatchNorm1d),
        ``'ln'`` (LayerNorm), or ``'none'``.

    Returns
    -------
    encoder:
        ``EncoderMultiMaskWrapper``-wrapped gene transformer encoder.
    proj_head:
        ``ProjectionHead`` MLP that maps pooled encoder outputs to the
        projection space.
    """
    # Build the encoder backbone (always api_version='v4' for multi-mask
    # wrapper support, which is needed to encode V views in one call)
    encoder = gt.__dict__["init_gt_encoder"](
        encoder_type=gt_type,
        n_special_values=n_special_values,
        count_encoding=count_encoding,
        n_value_bins=n_value_bins,
        cell_pos_enc=cell_pos_enc,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        embed_dim=enc_emb_dim,
        depth=enc_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        api_version="v4",
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        pos_learnable=pos_learnable,
        nz_spc=nz_spc,
        mlp_bias=mlp_bias,
    )
    encoder = EncoderMultiMaskWrapper(encoder)

    # Build the projection head
    proj_head = ProjectionHead(
        in_dim=enc_emb_dim,
        proj_dim=proj_dim,
        hidden_dim=proj_hidden_dim,
        norm=proj_norm,
    )

    # Weight initialisation (encoder)
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        _init_weights(m)

    encoder.to(device)
    proj_head.to(device)

    logger.info(encoder)
    logger.info(proj_head)

    def _count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"ST-JEPA encoder parameters: {_count_params(encoder):,}")
    logger.info(f"ST-JEPA projection head parameters: {_count_params(proj_head):,}")

    return encoder, proj_head


def init_stjepa_opt(
    encoder: nn.Module,
    proj_head: nn.Module,
    iterations_per_epoch: int,
    start_lr: float,
    ref_lr: float,
    warmup: int,
    num_epochs: int,
    wd: float = 1e-6,
    final_wd: float = 1e-6,
    final_lr: float = 0.0,
    use_bfloat16: bool = False,
    ipe_scale: float = 1.25,
) -> tuple[torch.optim.AdamW, None, WarmupCosineSchedule, CosineWDSchedule]:
    """
    Initialise the AdamW optimiser and cosine LR / WD schedulers for
    ST-JEPA training (encoder + projection head jointly).

    Parameters
    ----------
    encoder:
        The wrapped encoder module.
    proj_head:
        The projection head module.
    iterations_per_epoch, start_lr, ref_lr, warmup, num_epochs, wd,
    final_wd, final_lr, use_bfloat16, ipe_scale:
        Training schedule configuration parameters.

    Returns
    -------
    optimizer, scaler (always None for bfloat16), scheduler, wd_scheduler
    """
    # Separate weight-decayed params from bias / 1-D params
    param_groups = [
        {
            "params": [
                p
                for n, p in list(encoder.named_parameters())
                + list(proj_head.named_parameters())
                if ("bias" not in n) and (len(p.shape) != 1)
            ]
        },
        {
            "params": [
                p
                for n, p in list(encoder.named_parameters())
                + list(proj_head.named_parameters())
                if ("bias" in n) or (len(p.shape) == 1)
            ],
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    logger.info("ST-JEPA: initialising optimiser: AdamW.")
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    logger.info("ST-JEPA: initialising LR scheduler: WarmupCosineSchedule.")
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    logger.info("ST-JEPA: initialising WD scheduler: CosineWDSchedule.")
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    # GradScaler should not be used with bfloat16
    scaler = None

    return optimizer, scaler, scheduler, wd_scheduler


# ---------------------------------------------------------------------------
# Inference-oriented wrappers
# ---------------------------------------------------------------------------

def init_model(
    gt_type: str,
    count_encoding: str,
    n_value_bins: int,
    cell_pos_enc: str,
    device: str,
    vocab_size: int,
    seq_len: int,
    n_special_tokens: int,
    n_segments: int,
    n_special_values: int | None = None,
    enc_emb_dim: int = 768,
    enc_depth: int = 12,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    use_flash_attention: bool = True,
    use_layer_norm: bool = True,
    sep_gene_tokens_neb: bool = False,
    pos_learnable: bool = False,
    nz_spc: bool = False,
    mlp_bias: bool = False,
    proj_dim: int = 256,
    proj_hidden_dim: int | None = None,
    proj_norm: Literal["bn", "ln", "none"] = "bn",
) -> tuple[EncoderMultiMaskWrapper, "ProjectionHead"]:
    """
    High-level model factory used by inference scripts.

    Thin wrapper around :func:`init_stjepa_model` that keeps the call
    signature expected by existing inference code (``encoder, _ = init_model(...)``).

    Returns
    -------
    encoder:
        ``EncoderMultiMaskWrapper``-wrapped encoder.
    proj_head:
        ``ProjectionHead`` (not needed for inference but returned for
        checkpoint-loading convenience).
    """
    return init_stjepa_model(
        gt_type=gt_type,
        count_encoding=count_encoding,
        n_value_bins=n_value_bins,
        cell_pos_enc=cell_pos_enc,
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_special_tokens=n_special_tokens,
        n_segments=n_segments,
        proj_dim=proj_dim,
        n_special_values=n_special_values,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        proj_hidden_dim=proj_hidden_dim,
        proj_norm=proj_norm,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        use_layer_norm=use_layer_norm,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        pos_learnable=pos_learnable,
        nz_spc=nz_spc,
        mlp_bias=mlp_bias,
    )


def load_checkpoint(
    device: str | torch.device,
    r_path: str,
    encoder: nn.Module | None = None,
    opt: torch.optim.Optimizer | None = None,
    proj_head: nn.Module | None = None,
) -> tuple[nn.Module | None, int]:
    """
    Load an ST-JEPA checkpoint.

    The checkpoint dict is expected to contain at least the key
    ``"encoder"`` (state dict of the ``EncoderMultiMaskWrapper``).
    Optionally it may contain ``"proj_head"``, ``"opt"``, and
    ``"epoch"``.

    Handles the common case where the checkpoint was saved under DDP
    (state-dict keys prefixed with ``module.``): the prefix is stripped
    automatically if the target model is **not** wrapped in DDP.

    Parameters
    ----------
    device:
        Map location for ``torch.load``.
    r_path:
        Path to the ``.pth.tar`` / ``.pt`` checkpoint file.
    encoder:
        Encoder module to load weights into.
    opt:
        Optimizer to restore (optional, inference can pass ``None``).
    proj_head:
        Projection head to load weights into (optional).

    Returns
    -------
    encoder:
        The encoder with loaded weights (or ``None`` if not provided).
    epoch:
        The epoch stored in the checkpoint (``0`` if absent).
    """
    ckpt = torch.load(r_path, map_location=device)

    def _maybe_strip_ddp(state_dict: dict) -> dict:
        """Remove ``module.`` prefix added by DistributedDataParallel."""
        if any(k.startswith("module.") for k in state_dict):
            return {k.removeprefix("module."): v for k, v in state_dict.items()}
        return state_dict

    if encoder is not None and "encoder" in ckpt:
        encoder.load_state_dict(_maybe_strip_ddp(ckpt["encoder"]))
        logger.info(f"Loaded encoder weights from {r_path}")

    if proj_head is not None and "proj_head" in ckpt:
        proj_head.load_state_dict(_maybe_strip_ddp(ckpt["proj_head"]))
        logger.info(f"Loaded projection head weights from {r_path}")

    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])

    epoch = ckpt.get("epoch", 0)
    return encoder, epoch