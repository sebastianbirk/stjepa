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
    sample_segments:
        If ``True``, randomly sample the number of neighbor segments to
        keep each batch, matching the behaviour of ``BlockMaskCollator``.
    n_segments:
        Total number of segments (cell + neighbors).  Required when
        ``sample_segments=True``.
    seq_len_cell:
        Number of tokens per segment.  Required when
        ``sample_segments=True``.
    n_special_tokens:
        Number of special tokens prepended to the sequence.  Required
        when ``sample_segments=True``.
    cell_segment_sampling_ratio:
        Probability of keeping only the cell segment (k=1) when
        ``sample_segments=True``.  Matches ``BlockMaskCollator`` default.
    """

    def __init__(
        self,
        V: int,
        min_view_ratio: float = 0.2,
        max_view_ratio: float = 0.8,
        sample_segments: bool = False,
        n_segments: int | None = None,
        seq_len_cell: int | None = None,
        n_special_tokens: int = 0,
        cell_segment_sampling_ratio: float = 0.09090909090909091,
    ):
        assert 0 < min_view_ratio <= max_view_ratio <= 1.0, (
            "Require 0 < min_view_ratio <= max_view_ratio <= 1.0, "
            f"got {min_view_ratio} and {max_view_ratio}.")
        assert V >= 1, f"Need at least 1 view, got V={V}."
        if sample_segments:
            assert n_segments is not None and seq_len_cell is not None, (
                "n_segments and seq_len_cell are required when "
                "sample_segments=True.")

        self.V = V
        self.min_view_ratio = min_view_ratio
        self.max_view_ratio = max_view_ratio
        self.sample_segments = sample_segments
        self.n_segments = n_segments
        self.seq_len_cell = seq_len_cell
        self.n_special_tokens = n_special_tokens
        self.cell_segment_sampling_ratio = cell_segment_sampling_ratio
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

        collated = torch.utils.data.default_collate(batch)

        # ------------------------------------------------------------------
        # Segment sampling — zero out neighbor segments beyond sampled k
        # ------------------------------------------------------------------
        if self.sample_segments:
            if torch.rand(1).item() < self.cell_segment_sampling_ratio:
                k = 1
            else:
                k = torch.randint(
                    low=2, high=self.n_segments + 1, size=(1,)).item()
            cutoff = self.n_special_tokens + self.seq_len_cell * k

            for field, pad_val in (
                ("tokens",      0),
                ("segments",    0),
                ("positions",   0),
                ("values",      0.0),
                ("rel_x_coords", float("-inf")),
                ("rel_y_coords", float("-inf")),
            ):
                if field in collated:
                    collated[field][:, cutoff:] = pad_val

        # ------------------------------------------------------------------
        # Attention mask and view sampling (unchanged from here down)
        # ------------------------------------------------------------------
        tokens = collated["tokens"]   # [B, N]
        B = tokens.shape[0]

        masks_attention = (tokens != 0).unsqueeze(1).unsqueeze(1)  # [B,1,1,N]

        all_views: list[list[torch.Tensor]] = [
            self._sample_views(tokens[i]) for i in range(B)
        ]

        min_view_len: int = min(
            v.numel()
            for sample_views in all_views
            for v in sample_views
        )

        view_masks: list[torch.Tensor] = []
        for v_idx in range(self.V):
            view_v = torch.stack(
                [
                    all_views[b][v_idx][:min_view_len].to(torch.long)
                    for b in range(B)
                ],
                dim=0,
            )
            view_masks.append(view_v)

        return collated, view_masks, masks_attention