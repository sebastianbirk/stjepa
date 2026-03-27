"""
Adapted from Bardes, A et al. Revisiting Feature Prediction for Learning
Visual Representations from Video. arXiv:2404.08471 (2024);
https://github.com/facebookresearch/jepa/blob/main/src/models/utils/multimask.py
(25.03.2025).
"""

import torch
import torch.nn as nn


class EncoderMultiMaskWrapper(nn.Module):
    """
    EncoderMultiMaskWrapper class for encoding iteratively with multiple
    masks.

    In ST-JEPA this is used to encode V independent views in a single
    ``forward`` call: pass a list of V masks and receive a list of V
    embedding tensors back.

    Parameters
    ----------
    backbone:
        The encoder backbone network.
    """
    def __init__(self,
                 backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self,
                batch: dict[torch.Tensor],
                masks: torch.Tensor | list | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> tuple[list[torch.Tensor], dict[torch.Tensor]]:
        if masks is None:
            return self.backbone(batch=batch,
                                 masks=None,
                                 masks_attention=masks_attention)

        if isinstance(masks, torch.Tensor):
            masks = [masks]
        outs = []
        for m in masks:
            x, token_emb = self.backbone(batch=batch,
                                         masks=m,
                                         masks_attention=masks_attention)
            outs.append(x)
        return outs, token_emb
