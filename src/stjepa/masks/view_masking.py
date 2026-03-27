"""
ViewMaskCollator for ST-JEPA-style self-supervised learning on spatial
transcriptomics data.

Instead of sampling a context/target split (JEPA), this collator creates V
independent random *views* of each cell's gene expression profile.  Each view
is a random subset of the non-padded (non-zero) gene tokens, analogous to the
random-crop / colour-jitter augmentations used in image-based ST-JEPA.

Design rationale
----------------
Spatial transcriptomics tokens are stored as a flat integer sequence:

    [g_0, g_1, …, g_{N-1}]

where ``g_i = 0`` for absent genes (padding) and ``g_i > 0`` for expressed
genes.

For each cell we:
  1. Collect the global indices of non-zero gene tokens.
  2. Sample V random fractions (each drawn uniformly in
     [min_view_ratio, max_view_ratio]) of those indices.

All views within a batch are then trimmed to the minimum view length to
allow batch-stacking into a tensor of shape (V, B, min_view_len).
"""

import torch


class ViewMaskCollator:
    """
    Collator that samples V independent random gene-token views per cell.

    Parameters
    ----------
    V:
        Number of views to sample per cell.
    min_view_ratio:
        Minimum fraction of a cell's non-padded gene tokens to include in
        a single view.  Must be in (0, 1].
    max_view_ratio:
        Maximum fraction of a cell's non-padded gene tokens to include in
        a single view.  Must be in (0, 1] and >= min_view_ratio.
    """

    def __init__(
        self,
        V: int,
        min_view_ratio: float = 0.2,
        max_view_ratio: float = 0.8,
    ):
        assert 0 < min_view_ratio <= max_view_ratio <= 1.0, (
            "Require 0 < min_view_ratio <= max_view_ratio <= 1.0, "
            f"got {min_view_ratio} and {max_view_ratio}.")
        assert V >= 1, f"Need at least 1 view, got V={V}."

        self.V = V
        self.min_view_ratio = min_view_ratio
        self.max_view_ratio = max_view_ratio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_views(
        self,
        tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Sample V random gene-token views for a single cell.

        Parameters
        ----------
        tokens:
            1-D token sequence of shape (N,).  Gene tokens are zero
            where the gene is absent / padded.

        Returns
        -------
        views:
            List of V 1-D long tensors, each containing the *global*
            token indices (into the full sequence of length N) for that
            view.
        """
        # Identify non-padded gene token positions (global indices)
        nz_global = (tokens != 0).nonzero(as_tuple=False).squeeze(-1)
        K = nz_global.numel()

        if K == 0:
            # Degenerate case: all genes are zero (fully padded cell).
            # Return a single-element view with the first position
            # to avoid empty tensors downstream.
            fallback = torch.tensor([0], dtype=torch.long)
            return [fallback.clone() for _ in range(self.V)]

        views: list[torch.Tensor] = []
        for _ in range(self.V):
            # Sample a view ratio uniformly in [min, max]
            ratio = (
                self.min_view_ratio
                + torch.rand(1).item()
                * (self.max_view_ratio - self.min_view_ratio)
            )
            n_keep = max(1, round(K * ratio))
            # Random permutation and selection
            perm = torch.randperm(K)[:n_keep]
            view_idx = nz_global[perm]

            views.append(view_idx)

        return views

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(
        self,
        batch: list[dict],
    ) -> tuple[dict, list[torch.Tensor], torch.Tensor]:
        """
        Collate a batch and produce V view masks per sample.

        Parameters
        ----------
        batch:
            List of per-sample dictionaries as returned by the cell
            dataset.  Must contain at least ``'tokens'``; all other keys
            (``'positions'``, ``'segments'``, ``'values'``,
            ``'rel_x_coords'``, ``'rel_y_coords'``, …) are collated
            unchanged.

        Returns
        -------
        collated:
            Standard collated batch dictionary (all fields stacked along
            a new batch dimension).
        view_masks:
            List of V tensors, each of shape ``(B, min_view_len)``,
            containing the token indices for that view.  All views are
            trimmed to the minimum view length observed in the batch so
            that they can be stacked into a rectangular tensor.
            These masks can be passed directly to
            ``EncoderMultiMaskWrapper`` (which expects a list of masks,
            one per forward pass).
        masks_attention:
            Boolean attention mask derived from the *full* (unmasked)
            token sequence, shape ``(B, 1, 1, N)``.  A position is
            ``True`` where the token is non-zero.  Pass this to the
            encoder so that padded positions in the full sequence are
            never attended to.
        """
        collated = torch.utils.data.default_collate(batch)
        tokens = collated["tokens"]   # [B, N]
        B = tokens.shape[0]

        # Full-sequence attention mask (True = valid, False = padding)
        masks_attention = (tokens != 0).unsqueeze(1).unsqueeze(1)  # [B,1,1,N]

        # Sample V views for every sample in the batch
        all_views: list[list[torch.Tensor]] = [
            self._sample_views(tokens[i]) for i in range(B)
        ]

        # Trim to the global minimum view length so we can stack into a
        # rectangular tensor without padding
        min_view_len: int = min(
            v.numel()
            for sample_views in all_views
            for v in sample_views
        )

        # Build one [B, min_view_len] tensor per view
        view_masks: list[torch.Tensor] = []
        for v_idx in range(self.V):
            view_v = torch.stack(
                [
                    all_views[b][v_idx][:min_view_len].to(torch.long)
                    for b in range(B)
                ],
                dim=0,
            )  # [B, min_view_len]
            view_masks.append(view_v)

        return collated, view_masks, masks_attention
