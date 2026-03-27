"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
(05.06.2024).
"""

import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, Literal, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .modules import Attention, Block, MLP, ValueEmbWeightsProjection
from .utils import (get_1d_sincos_pos_embed,
                    get_1d_sincos_pos_embed_from_coord,
                    trunc_normal_)
from ..masks.utils import apply_masks

DEBUG = False


class GeneTransformerBaseEncoder(ABC, nn.Module):
    """
    GeneTransformerBaseEncoder class to encode contexts or targets.

    Parameters
    -----------
    vocab_size:
        Size of the token vocabulary. Includes <pad> token.
    seq_len:
        Length of the token sequences.
    n_segments:
        Number of token segments within a token sequence.
    cell_pos_enc:
        Cell position encoding. Either `segment` if cells are ranked
        based on distance to index cell or `coords` if relative
        positions to index cell are used.
    embed_dim:
        Dimension of the encoder embedding.
    depth:
        Number of transformer blocks in the encoder.
    predictor_embed_dim:
        Dimension of the predictor embedding.
    predictor_depth:
        Number of transformer blocks in the predictor.
    num_heads:
        Number of attention heads in the Attention modules.
    mlp_ratio:
        Ratio to determine number of hidden dimensions in MLP modules
        compared to input and output dimensions.
    qkv_bias:
        If `True`, include bias in query, key, and value layers of
        Attention modules.
    qk_scale:
        Scaling factor for query and key vectors of Attention modules.
    drop_rate:
        Dropout ratio in projection layer of Attention modules and in
        layers of MLP modules.
    attn_drop_rate:
        Dropout ratio in attention layer of Attention modules.
    norm_layer:
        Normalization layer.
    init_std:
        Standard deviation for weight initialization.
    use_flash_attention:
        If `True`, use flash_attention.
    sep_gene_tokens_neb:
        If `True`, use separate gene tokens for neighborhood.
    """
    def __init__(self,
                 vocab_size: int,
                 seq_len: int,
                 n_segments: int,
                 cell_pos_enc: Literal['segment', 'coord'],
                 embed_dim: int = 768,
                 depth: int = 12,
                 predictor_embed_dim: int = 384,
                 predictor_depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: nn.modules.normalization=nn.LayerNorm,
                 init_std: float = 0.02,
                 use_flash_attention: bool = True,
                 sep_gene_tokens_neb: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.seq_len_cell = seq_len // n_segments
        self.cell_pos_enc = cell_pos_enc
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.sep_gene_tokens_neb = sep_gene_tokens_neb

        # Initialize token embeddings
        self.token_embed = nn.Embedding(
            vocab_size + (vocab_size if sep_gene_tokens_neb else 0), # already includes <pad>
            embed_dim,
            padding_idx=0)

        # Initialize segment embeddings
        if self.cell_pos_enc == 'segment':
            self.seg_embed = nn.Embedding(
                1 + n_segments,  # include <pad>
                embed_dim,
                padding_idx=0)

            # Prevent gradient updates and initialize with sincos embedding
            self.seg_embed.weight.requires_grad = False
            seg_embed = get_1d_sincos_pos_embed(
                embed_dim=embed_dim,
                n_zero_pos=0,
                n_sincos_pos=n_segments)
            self.seg_embed.weight[1:].copy_(torch.from_numpy(seg_embed).float())

        # Initialize encoder blocks and norm layer
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  act_layer=nn.GELU,
                  attn_drop=attn_drop_rate,
                  norm_layer=norm_layer,
                  use_flash_attention=use_flash_attention)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize weights of layers
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Compute omega for segment embedding: 1 / 10000^{2i/dim}
        if self.cell_pos_enc == 'coord':
            self.coord_omega = torch.arange(
                embed_dim // 4, dtype=torch.float32)
            self.coord_omega = 1.0 / (
                10000 ** (self.coord_omega / (embed_dim / 4)))

    def _rescale_blocks(self):
        """
        Helper function to scale initialized layer weights.
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """
        Helper function to initialize layer weights.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _compute_layer_emb(
            self,
            x: torch.Tensor,
            attn_base: torch.Tensor | None,
            layers: Sequence[int],
            masks: list[torch.Tensor] | torch.Tensor | None,
            cell_only: bool) -> dict[int, torch.Tensor]:
        """
        Helper function to return embeddings for either full context or
        cell only context.
        """
        layers: list[int] = sorted({int(l) for l in layers})
        max_layer: int = max(layers)

        # Format masks
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        attn = None
        if attn_base is not None:
            if attn_base.size(2) == 1: # [B, 1, 1, L] -> [B, 1, L, L]
                attn = attn_base.expand(-1, -1, attn_base.size(-1), -1).clone()
            else:
                attn = attn_base.clone()
            # Optionally block cross-attention from cell queries to
            # neighborhood keys
            if cell_only:
                attn[
                    :,
                    :,
                    :self.seq_len_cell,
                    self.seq_len_cell:] = False

        if cell_only:
            x[:, self.seq_len_cell:, :] = 0

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)

        if DEBUG:
            zero_rows = (x.abs().sum(dim=-1) == 0)   # (B, S)
            percentage_zero_rows = zero_rows.float().mean()
            print("Fraction of preblock X all-zero rows:", percentage_zero_rows)

        # Run forward prop and store embeddings for each specified layer
        out: dict[int, torch.Tensor] = {}
        if DEBUG:
            valid_mask = x.ne(0).any(dim=-1)
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x, masks=attn)
            if i == len(self.blocks) and (self.norm is not None):
                # Apply norm only for last layer as in training
                x = self.norm(x)
                if DEBUG:
                    x = x * valid_mask.unsqueeze(-1)
                    zero_rows = (x.abs().sum(dim=-1) == 0)   # (B, S)

                    percentage_zero_rows = zero_rows.float().mean()
                    print("Fraction of afterblock X all-zero rows:", percentage_zero_rows)
            if i in layers:
                out[i] = x
            if i == max_layer:
                break

        return out

    def _get_seg_emb(
            self,
            batch: dict):
        """
        Helper function to get segment embeddings.
        """
        if self.cell_pos_enc == 'segment':
            seg_emb = self.seg_embed(batch['segments'])
        elif self.cell_pos_enc == 'coord':
            rel_x_coord_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.embed_dim // 2,
                omega=self.coord_omega,
                coord=batch['rel_x_coords'])
            rel_y_coord_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.embed_dim // 2,
                omega=self.coord_omega,
                coord=batch['rel_y_coords'])
            seg_emb = torch.cat(
                [rel_x_coord_emb, rel_y_coord_emb], dim=-1)
        else:
            raise ValueError(
                f"Unknown cell_pos_enc: {self.cell_pos_enc}.")

        return seg_emb

    @torch.no_grad()
    def return_token_emb(self,
                         tokens: torch.Tensor,
                         ) -> torch.Tensor:
        """
        Return the token embeddings for a batch of input tokens.
        """
        return self.token_embed(tokens)

    @torch.no_grad()
    def return_seg_emb(self,
                       segments: torch.Tensor,
                       ) -> torch.Tensor:
        """
        Return the segment embeddings for a batch of input segments.
        """
        return self.seg_embed(segments)

    @abstractmethod
    def forward(self) -> torch.Tensor:
        pass

    @abstractmethod
    def return_layer_emb() -> tuple[
            dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        pass




class GeneTransformerRankEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerRankEncoder class to encode contexts or targets using
    ranks based on gene expression counts.
    """
    def __init__(self,
                 pos_learnable: bool = False,
                 **base_encoder_kwargs,
                 ):
        super().__init__(**base_encoder_kwargs)

        self.pos_embed = nn.Embedding(self.seq_len + 1,
                                      self.embed_dim,
                                      padding_idx=0)

        if not pos_learnable:
            self.pos_embed.weight.requires_grad = False
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                n_zero_pos=0,
                n_sincos_pos=self.seq_len)
            self.pos_embed.weight[1:].copy_(
                torch.from_numpy(pos_embed).float())

    def forward(self,
                batch: dict[torch.Tensor],
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict]:
            if masks is not None:
                if not isinstance(masks, list):
                    masks = [masks]

            pos_emb = self.pos_embed(batch['positions'])
            seg_emb = self._get_seg_emb(batch)
            token_emb = self.token_embed(batch['tokens'])

            x = seg_emb + pos_emb + token_emb

            if masks is not None:
                x = apply_masks(x, masks)

            for i, blk in enumerate(self.blocks):
                x = blk(x, masks=masks_attention)
            if self.norm is not None:
                x = self.norm(x)

            return x, token_emb

    @torch.inference_mode()
    def return_layer_emb(
            self,
            layers: Sequence[int],
            batch: Mapping[str, torch.Tensor],
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            need_cell_only_context: bool = True,
            ) -> tuple[
                dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        if not layers:
            raise ValueError(
                "Layers must be a non-empty sequence of positive integers.")

        token_emb = self.token_embed(batch["tokens"])
        pos_emb = self.pos_embed(batch['positions'])
        seg_emb = self._get_seg_emb(batch)

        x = seg_emb + pos_emb + token_emb

        full_ctx = self._compute_layer_emb(
            x, masks_attention, layers, masks, cell_only=False)
        cell_only_ctx = self._compute_layer_emb(
            x, masks_attention, layers, masks,
            cell_only=True) if need_cell_only_context else None

        return full_ctx, cell_only_ctx


class GeneTransformerCountEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerCountEncoder class to encode contexts or targets
    using gene expression counts.
    """
    def __init__(self,
                 count_encoding: Literal['value_bins', 'mlp'] = 'mlp',
                 mlp_bias: bool = True,
                 n_value_bins: int | None = 100,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)
        self.count_encoding = count_encoding
        self.mlp_bias = mlp_bias
        self.n_value_bins = n_value_bins

        if self.count_encoding == 'value_bins':
            self.value_embed = nn.Embedding(
                self.n_value_bins,
                self.embed_dim)
            self.value_emb_weights_projection = ValueEmbWeightsProjection(
                dim=self.n_value_bins)
        elif self.count_encoding == 'mlp':
            hidden_dim = int(self.embed_dim/2)
            self.value_embed = MLP(
                in_features=1,
                hidden_features=hidden_dim,
                out_features=self.embed_dim,
                bias=self.mlp_bias,
                act_layer=nn.GELU)

    def _get_value_emb(self, batch: dict) -> torch.Tensor:
        """Compute value embeddings for gene tokens."""
        if self.count_encoding == 'value_bins':
            value_emb_weights = self.value_emb_weights_projection(
                batch['values'].unsqueeze(-1))
            value_emb = value_emb_weights @ self.value_embed.weight

            zero_counts_mask = batch['values'] == 0.0
            value_emb[zero_counts_mask] = 0.0

        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(batch['values'].unsqueeze(-1))
        else:
            raise ValueError(
                f"Unknown count_encoding: {self.count_encoding}.")
        return value_emb

    def forward(self,
                batch: dict[torch.Tensor],
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict]:
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        seg_emb = self._get_seg_emb(batch)
        token_emb = self.token_embed(batch['tokens'])
        value_emb = self._get_value_emb(batch)

        x = seg_emb + token_emb + value_emb

        if masks is not None:
            x = apply_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=masks_attention)
        if self.norm is not None:
            x = self.norm(x)

        return x, token_emb

    @torch.inference_mode()
    def return_layer_emb(
            self,
            layers: Sequence[int],
            batch: Mapping[str, torch.Tensor],
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            need_cell_only_context: bool = True,
            ) -> tuple[
                dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        if not layers:
            raise ValueError(
                "Layers must be a non-empty sequence of positive integers.")

        token_emb = self.token_embed(batch['tokens'])
        seg_emb = self._get_seg_emb(batch)
        value_emb = self._get_value_emb(batch)

        x = seg_emb + token_emb + value_emb

        full_ctx = self._compute_layer_emb(
            x, masks_attention, layers, masks, cell_only=False)
        cell_only_ctx = self._compute_layer_emb(
            x, masks_attention, layers, masks,
            cell_only=True) if need_cell_only_context else None

        return full_ctx, cell_only_ctx


class GeneTransformerCombinedEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerCombinedEncoder class to encode contexts or targets
    using gene tokens and gene expression counts.
    """
    def __init__(self,
                 count_encoding: Literal['value_bins', 'mlp'] = 'mlp',
                 mlp_bias: bool = True,
                 n_value_bins: int | None = 100,
                 pos_learnable: bool = False,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)
        self.count_encoding = count_encoding
        self.mlp_bias = mlp_bias
        self.n_value_bins = n_value_bins

        self.pos_embed = nn.Embedding(self.seq_len + 1,
                                      self.embed_dim,
                                      padding_idx=0)

        if not pos_learnable:
            self.pos_embed.weight.requires_grad = False
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                n_zero_pos=0,
                n_sincos_pos=self.seq_len)
            self.pos_embed.weight[1:].copy_(
                torch.from_numpy(pos_embed).float())

        if self.count_encoding == 'value_bins':
            self.value_embed = nn.Embedding(
                self.n_value_bins,
                self.embed_dim)
            self.value_emb_weights_projection = ValueEmbWeightsProjection(
                dim=self.n_value_bins)
        elif self.count_encoding == 'mlp':
            hidden_dim = int(self.embed_dim/2)
            self.value_embed = MLP(
                in_features=1,
                hidden_features=hidden_dim,
                out_features=self.embed_dim,
                bias=self.mlp_bias,
                act_layer=nn.GELU)

    def _get_value_emb(self, batch: dict) -> torch.Tensor:
        """Compute value embeddings for gene tokens."""
        if self.count_encoding == 'value_bins':
            value_emb_weights = self.value_emb_weights_projection(
                batch['values'].unsqueeze(-1))
            value_emb = value_emb_weights @ self.value_embed.weight

            zero_counts_mask = batch['values'] == 0.0
            value_emb[zero_counts_mask] = 0.0

        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(batch['values'].unsqueeze(-1))
        else:
            raise ValueError(
                f"Unknown count_encoding: {self.count_encoding}.")
        return value_emb

    def forward(self,
                batch: dict[torch.Tensor],
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict]:
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        pos_emb = self.pos_embed(batch['positions'])
        seg_emb = self._get_seg_emb(batch)
        token_emb = self.token_embed(batch['tokens'])
        value_emb = self._get_value_emb(batch)

        x = seg_emb + pos_emb + token_emb + value_emb

        if masks is not None:
            x = apply_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=masks_attention)
        if self.norm is not None:
            x = self.norm(x)

        return x, token_emb

    @torch.inference_mode()
    def return_layer_emb(
            self,
            layers: Sequence[int],
            batch: Mapping[str, torch.Tensor],
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            need_cell_only_context: bool = True,
            ) -> tuple[
                dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        if not layers:
            raise ValueError(
                "Layers must be a non-empty sequence of positive integers.")

        token_emb = self.token_embed(batch['tokens'])
        pos_emb = self.pos_embed(batch['positions'])
        seg_emb = self._get_seg_emb(batch)

        if DEBUG:
            proportion_zero = (batch['tokens'] == 0).float().mean()
            print("Proportion of zero tokens:", proportion_zero)

        value_emb = self._get_value_emb(batch)

        x = seg_emb + pos_emb + token_emb + value_emb

        full_ctx = self._compute_layer_emb(
            x, masks_attention, layers, masks, cell_only=False)
        cell_only_ctx = self._compute_layer_emb(
            x, masks_attention, layers, masks,
            cell_only=True) if need_cell_only_context else None

        return full_ctx, cell_only_ctx


def init_gt_encoder(
        encoder_type: Literal['rank', 'counts', 'combined'],
        **encoder_kwargs
        ) -> GeneTransformerRankEncoder | GeneTransformerCountEncoder | GeneTransformerCombinedEncoder:
    """
    Initialize GeneTransformerEncoder based on encoder type.
    """
    if encoder_type == 'rank':
        model = GeneTransformerRankEncoder(
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)
    elif encoder_type == 'counts':
        model = GeneTransformerCountEncoder(
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)
    elif encoder_type == 'combined':
        model = GeneTransformerCombinedEncoder(
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)

    return model
