"""
Simple collator for inference that batches data and creates a padding-aware
attention mask.  No view or block masking is applied — the full token
sequence is passed to the encoder.
"""

import torch


class InferenceCollator:
    """
    Collator for inference / embedding extraction.

    Returns
    -------
    collated : dict[str, Tensor]
        Standard collated batch dictionary.
    masks_attention : Tensor
        Boolean attention mask of shape ``(B, 1, 1, N)`` where ``True``
        indicates a valid (non-padding) token position.
    """

    def __call__(
        self,
        batch: list[dict],
    ) -> tuple[dict, torch.Tensor]:
        collated = torch.utils.data.default_collate(batch)
        tokens = collated["tokens"]  # [B, N]

        # True where token is non-zero (valid); False for padding
        masks_attention = (tokens != 0).unsqueeze(1).unsqueeze(1)  # [B,1,1,N]

        return collated, masks_attention
