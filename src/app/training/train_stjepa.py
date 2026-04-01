"""
ST-JEPA training loop for spatial transcriptomics data.

Adapts the architecture from:
    Quillent, A., Bourdoukan, R. & Löwe, S.  LeJEPA: Provable and Scalable
    Self-Supervised Learning Without the Heuristics. arXiv:2502.01048 (2025).

Key differences from the standard NicheJEPA JEPA training loop
(train.py):

  * **No predictor, no target / EMA encoder.**  ST-JEPA is a single-encoder
    method; only an online encoder and a lightweight projection head are
    trained.
  * **V random views instead of context/target split.**  Each cell's
    non-padded gene tokens are randomly subsampled into V independent views
    (handled by ``ViewMaskCollator``).  The encoder encodes all V views and
    the projection head maps each pooled representation to the projection
    space.
  * **ST-JEPA loss = invariance loss + λ · SIGReg.**
    - *Invariance loss*: encourages all V projections of the same cell to
      agree (``(proj.mean(0) - proj)²``).
    - *SIGReg*: prevents collapse by penalising the deviation of the
      empirical characteristic function of projections from a Gaussian CF.
  * **Adversarial batch-effect removal (optional).**
    When ``adversarial_labels`` is non-empty, adversarial classification
    heads with gradient reversal are attached to the mean-pooled gene-token
    representations.  The adversarial cross-entropy loss is weighted by
    ``adv_weight`` and added to the total loss.  Setting ``adv_weight=0``
    disables the adversarial loss.

Usage
-----
This function is called from ``main_stjepa.py`` and accepts a parameter dict
loaded from a YAML config file.  See ``configs/model/example/stjepa.yaml``
for a fully-annotated example.
"""

import logging
import os
import sys
import yaml
from datetime import datetime

import datasets
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from app.utils import init_stjepa_model, init_stjepa_opt
from stjepa.datasets.cell_datasets import CellDataset
from stjepa.datasets.dataloaders import init_dataloader_and_sampler
from stjepa.masks.view_masking import ViewMaskCollator
from stjepa.models.adversarial import (AdversarialHead,
                                        adversarial_lambda_schedule)
from stjepa.models.sigreg import SIGReg
from stjepa.utils.distributed import init_distributed
from stjepa.utils.logging import AverageMeter, grad_logger

import pickle

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

_GLOBAL_SEED = 0

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

def _pool(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Reduce per-token encoder output ``[B, L, D]`` to ``[B, D]``.

    Parameters
    ----------
    x:
        Encoder output for a single view, shape ``(B, L, D)``.
    mode:
        ``'mean'``: mean-pool over all L positions.

    Returns
    -------
    pooled:
        Shape ``(B, D)``.
    """
    return x.mean(dim=1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stjepa(
    args: dict,
    train_dataset: datasets.Dataset,
    resume_preempt: bool = False,
    save_folder_path: str | None = None,
    LOCAL_RANK: int | None = None,
    WORLD_RANK: int | None = None,
):
    """
    Train a ST-JEPA model on spatial transcriptomics data.

    Parameters
    ----------
    args:
        Nested dict loaded from the YAML config file.
    train_dataset:
        HuggingFace Dataset (or list of Datasets for multi-epoch
        precomputed splits) containing the tokenised cells.
    resume_preempt:
        If ``True``, resume from the latest checkpoint in
        ``save_folder_path``.
    save_folder_path:
        Directory to write checkpoints and config dumps.
    LOCAL_RANK:
        Per-node GPU index (set by the launcher).
    WORLD_RANK:
        Global process rank (set by the launcher).
    """
    # ------------------------------------------------------------------
    # Reproducibility and precision
    # ------------------------------------------------------------------
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif LOCAL_RANK is not None:
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # ------------------------------------------------------------------
    # Config parsing – data
    # ------------------------------------------------------------------
    dataset_name          = args["data"]["dataset_name"]
    token_dict_folder_path = args["data"]["token_dict_folder_path"]
    seq_len_cell          = args["data"]["seq_len_cell"]
    seq_len_neighborhood  = args["data"]["seq_len_neighborhood"]
    n_segments            = args["data"]["n_segments"]
    sampling_strategy     = args["data"]["sampling_strategy"]
    batch_size            = args["data"]["batch_size"]
    num_workers           = args["data"]["num_workers"]
    pin_memory            = args["data"]["pin_memory"]
    sep_gene_tokens_neb   = args["data"].get("sep_gene_tokens_neb", False)
    use_sampler           = args["data"].get("use_sampler", False)
    mega_batch_mult_max   = args["data"].get("mega_batch_mult_max", 1000)

    # Config parsing – meta / model
    gt_type             = args["meta"]["gt_type"]
    count_encoding      = args["meta"]["count_encoding"]
    n_value_bins        = args["meta"]["n_value_bins"]
    cell_pos_enc        = args["meta"].get("cell_pos_enc", "segment")
    enc_depth           = args["meta"]["enc_depth"]
    enc_emb_dim         = args["meta"]["enc_emb_dim"]
    num_heads           = args["meta"].get("num_heads", 8)
    mlp_ratio           = args["meta"].get("mlp_ratio", 4.0)
    use_bfloat16        = args["meta"]["use_bfloat16"]
    use_flash_attention = args["meta"]["use_flash_attention"]
    pos_learnable       = args["meta"].get("pos_learnable", False)
    mlp_bias            = args["meta"].get("mlp_bias", False)

    # Config parsing – ST-JEPA-specific
    V               = args["stjepa"]["V"]
    proj_dim        = args["stjepa"]["proj_dim"]
    lamb            = args["stjepa"]["lamb"]
    pooling         = args["stjepa"].get("pooling", "mean")
    min_view_ratio  = args["stjepa"].get("min_view_ratio", 0.2)
    max_view_ratio  = args["stjepa"].get("max_view_ratio", 0.8)
    proj_hidden_dim = args["stjepa"].get("proj_hidden_dim", None)
    proj_norm       = args["stjepa"].get("proj_norm", "bn")
    n_sigreg_knots  = args["stjepa"].get("n_sigreg_knots", 17)
    n_sigreg_proj   = args["stjepa"].get("n_sigreg_proj", 256)

    # Config parsing – adversarial
    adv_cfg             = args.get("adversarial", {})
    adversarial_labels  = adv_cfg.get("labels", [])
    adv_weight          = adv_cfg.get("weight", 0.0)
    adv_schedule_gamma  = adv_cfg.get("schedule_gamma", 10.0)
    adv_hidden_dim      = adv_cfg.get("hidden_dim", None)
    adv_disc_update_freq = adv_cfg.get("disc_update_freq", 1)

    # Config parsing – optimisation
    warmup       = args["optimization"]["warmup"]
    num_epochs   = args["optimization"]["epochs"]
    start_lr     = args["optimization"]["start_lr"]
    lr           = args["optimization"]["lr"]
    final_lr     = args["optimization"]["final_lr"]
    wd           = float(args["optimization"]["weight_decay"])
    final_wd     = float(args["optimization"]["final_weight_decay"])
    ipe_scale    = args["optimization"]["ipe_scale"]
    clip_grad    = args["optimization"]["clip_grad"]

    # Config parsing – state / logging
    log_freq            = args["state"]["log_freq"]
    checkpoint_freq     = args["state"]["checkpoint_freq"]
    checkpoint_freq_iter = args["state"]["checkpoint_freq_iter"]
    write_tag           = args["state"]["write_tag"]
    load_model          = args["state"]["load_checkpoint"] or resume_preempt
    r_file              = args["state"]["read_checkpoint"]
    load_folder_path    = args["state"]["folder_path"]

    # Pre-computed non-zero token counts
    if "precomputed_epoch_n_nonzero_tokens" in args["data"]:
        with open(args["data"]["precomputed_epoch_n_nonzero_tokens"], "rb") as f:
            epoch_n_nonzero_tokens = pickle.load(f)
    elif args["data"].get("precomputed_n_nonzero_tokens"):
        with open(args["data"]["precomputed_n_nonzero_tokens"], "rb") as f:
            n_nonzero_tokens = pickle.load(f)
    else:
        n_nonzero_tokens = None

    # ------------------------------------------------------------------
    # Token dictionary
    # ------------------------------------------------------------------
    with open(token_dict_folder_path, "rb") as fh:
        token_dict = pickle.load(fh)
    vocab_size = len(token_dict)

    seq_len = seq_len_cell + seq_len_neighborhood

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    try:
        mp.set_start_method("spawn")
    except Exception:
        logger.info("Multiprocessing start method not changed.")

    world_size, rank = init_distributed()
    logger.info(f"Distributed (rank/world_size): {rank}/{world_size}.")
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # ------------------------------------------------------------------
    # Artifact folder
    # ------------------------------------------------------------------
    if not save_folder_path:
        artifact_root = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
            "artifacts")
        ts = (datetime.now().strftime("%d%m%Y_%H%M%S")
              + f"_{datetime.now().microsecond // 1000:03d}")
        save_folder_path = os.path.join(artifact_root, dataset_name, ts)
    if rank == 0:
        os.makedirs(save_folder_path, exist_ok=True)

    if rank == 0:
        dump = os.path.join(save_folder_path, "params.yaml")
        with open(dump, "w") as f:
            yaml.dump(args, f)

    log_file  = os.path.join(save_folder_path, f"{write_tag}_r{rank}.csv")
    save_path = os.path.join(save_folder_path,
                             f"{write_tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(save_folder_path, f"{write_tag}-latest.pth.tar")
    load_path = None
    if load_model:
        load_path = (
            os.path.join(load_folder_path, r_file)
            if r_file is not None
            else latest_path
        )

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    encoder, proj_head = init_stjepa_model(
        gt_type=gt_type,
        count_encoding=count_encoding,
        n_value_bins=n_value_bins,
        cell_pos_enc=cell_pos_enc,
        device=device,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_segments=n_segments,
        proj_dim=proj_dim,
        enc_emb_dim=enc_emb_dim,
        enc_depth=enc_depth,
        proj_hidden_dim=proj_hidden_dim,
        proj_norm=proj_norm,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        use_flash_attention=use_flash_attention,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        pos_learnable=pos_learnable,
        mlp_bias=mlp_bias,
    )

    # SIGReg loss module (buffers are on CPU; forward uses proj.device)
    sigreg = SIGReg(knots=n_sigreg_knots, n_projections=n_sigreg_proj).to(device)

    # ------------------------------------------------------------------
    # Adversarial heads (one per label)
    # ------------------------------------------------------------------
    adv_heads: dict[str, AdversarialHead] = {}
    adv_label_map: dict[str, dict[int, int]] = {}
    use_adversarial = adv_weight > 0 and len(adversarial_labels) > 0
    if use_adversarial:
        raw_label_map = adv_cfg.get("label_map", {})
        for label_name in adversarial_labels:
            if label_name not in raw_label_map:
                raise ValueError(
                    f"adversarial label '{label_name}' has no entry in "
                    f"adversarial.label_map. Please provide an explicit mapping, "
                    f"e.g.:\n"
                    f"  adversarial:\n"
                    f"    label_map:\n"
                    f"      {label_name}:\n"
                    f"        102: 0\n"
                    f"        103: 1"
                )
            mapping = {int(k): int(v) for k, v in raw_label_map[label_name].items()}
            expected = set(range(max(mapping.values()) + 1))
            if set(mapping.values()) != expected:
                raise ValueError(
                    f"label_map for '{label_name}' must map to contiguous "
                    f"integers starting from 0, got {sorted(set(mapping.values()))}."
                )
            adv_label_map[label_name] = mapping

        # Derive n_classes from label_map instead of requiring it in config
        adv_n_classes = {
            label_name: max(m.values()) + 1
            for label_name, m in adv_label_map.items()
        }

        for label_name in adversarial_labels:
            n_cls = adv_n_classes[label_name]
            head = AdversarialHead(
                in_dim=enc_emb_dim,
                n_classes=n_cls,
                hidden_dim=adv_hidden_dim,
            ).to(device)
            adv_heads[label_name] = head
            logger.info(
                f"Adversarial head for '{label_name}': "
                f"{sum(p.numel() for p in head.parameters()):,} params, "
                f"{n_cls} classes.")

    # ------------------------------------------------------------------
    # View mask collator
    # ------------------------------------------------------------------
    view_collator = ViewMaskCollator(
        V=V,
        min_view_ratio=min_view_ratio,
        max_view_ratio=max_view_ratio,
    )

    # ------------------------------------------------------------------
    # Dataset and dataloader
    # ------------------------------------------------------------------
    if isinstance(train_dataset, list):
        train_cell_datasets = []
        for d, nz in zip(train_dataset, epoch_n_nonzero_tokens):
            cell_d = CellDataset(
                dataset=d,
                vocab_size=vocab_size,
                seq_len_cell=seq_len_cell,
                seq_len_neighborhood=seq_len_neighborhood,
                gt_type=gt_type,
                cell_pos_enc=cell_pos_enc,
                adversarial_labels=adversarial_labels,
                sampling_strategy=sampling_strategy,
                n_nonzero_tokens_list=nz,
                include_cell_id=False,
                sep_gene_tokens_neb=sep_gene_tokens_neb,
            )
            train_cell_datasets.append(cell_d)
    else:
        train_cell_dataset = CellDataset(
            dataset=train_dataset,
            vocab_size=vocab_size,
            seq_len_cell=seq_len_cell,
            seq_len_neighborhood=seq_len_neighborhood,
            gt_type=gt_type,
            cell_pos_enc=cell_pos_enc,
            adversarial_labels=adversarial_labels,
            sampling_strategy=sampling_strategy,
            n_nonzero_tokens_list=n_nonzero_tokens,
            include_cell_id=False,
            sep_gene_tokens_neb=sep_gene_tokens_neb,
        )

    if isinstance(train_dataset, list):
        train_loaders, train_samplers = [], []
        for cell_d in train_cell_datasets:
            tl, ts = init_dataloader_and_sampler(
                cell_dataset=cell_d,
                batch_size=batch_size,
                use_length_grouped_sampler=use_sampler,
                world_size=world_size,
                rank=rank,
                collate_fn=view_collator,
                pin_memory=pin_memory,
                num_workers=num_workers,
                drop_last=True,
                prefetch_factor=(4 if num_workers > 0 else None),
                persistent_workers=False,
                mega_batch_mult_max=mega_batch_mult_max,
            )
            train_loaders.append(tl)
            train_samplers.append(ts)
    else:
        train_loader, train_sampler = init_dataloader_and_sampler(
            cell_dataset=train_cell_dataset,
            batch_size=batch_size,
            use_length_grouped_sampler=use_sampler,
            world_size=world_size,
            rank=rank,
            collate_fn=view_collator,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=True,
            prefetch_factor=(4 if num_workers > 0 else None),
            persistent_workers=False,
            mega_batch_mult_max=mega_batch_mult_max,
        )

    if isinstance(train_dataset, list):
        ipe = len(train_loaders[0])
    else:
        ipe = len(train_loader)
    print(f"Iterations per epoch: {ipe}.")

    total_steps = num_epochs * ipe

    # ------------------------------------------------------------------
    # Optimiser and schedulers
    # ------------------------------------------------------------------
    extra_params = list(adv_heads.values()) if use_adversarial else None
    optimizer, scaler, scheduler, wd_scheduler = init_stjepa_opt(
        encoder=encoder,
        proj_head=proj_head,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
        extra_params=extra_params,
    )

    # ------------------------------------------------------------------
    # DDP wrapping
    # ------------------------------------------------------------------
    # Convert BatchNorm in projection head to SyncBatchNorm for
    # consistent batch statistics across all GPUs
    proj_head = nn.SyncBatchNorm.convert_sync_batchnorm(proj_head)

    encoder = DistributedDataParallel(
        encoder,
        static_graph=True,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    proj_head = DistributedDataParallel(
        proj_head,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    # DDP-wrap adversarial heads
    ddp_adv_heads: dict[str, DistributedDataParallel] = {}
    for label_name, head in adv_heads.items():
        ddp_adv_heads[label_name] = DistributedDataParallel(
            head,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    start_epoch = 0
    if load_model and load_path is not None:
        try:
            ckpt = torch.load(load_path, map_location=device)
            start_epoch = ckpt.get("epoch", 0) + 1
            encoder.load_state_dict(ckpt["encoder"])
            proj_head.load_state_dict(ckpt["proj_head"])
            optimizer.load_state_dict(ckpt["opt"])
            # Restore adversarial heads if present
            for label_name in adv_heads:
                key = f"adv_head_{label_name}"
                if key in ckpt:
                    ddp_adv_heads[label_name].load_state_dict(ckpt[key])
            logger.info(f"Resumed from checkpoint: {load_path} (epoch {start_epoch}).")
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
            del ckpt
        except Exception as exc:
            logger.info(f"Could not load checkpoint: {exc}. Starting fresh.")
            start_epoch = 0

    # ------------------------------------------------------------------
    # Checkpoint saving helper
    # ------------------------------------------------------------------
    def save_checkpoint(epoch: int, itr: int | None = None):
        save_dict = {
            "encoder":   encoder.state_dict(),
            "proj_head": proj_head.state_dict(),
            "opt":       optimizer.state_dict(),
            "epoch":     epoch,
            "loss":      loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr":         lr,
        }
        # Save adversarial heads
        for label_name, ddp_head in ddp_adv_heads.items():
            save_dict[f"adv_head_{label_name}"] = ddp_head.state_dict()
        if itr is not None:
            save_dict["iter_number"] = itr
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                tag = f"{epoch}" if itr is None else f"{epoch}_{itr}"
                torch.save(save_dict, save_path.format(epoch=tag))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch}")

        if isinstance(train_dataset, list):
            logger.info(f"Using train loader for epoch {epoch}.")
            train_loader = train_loaders[epoch]
            train_sampler = train_samplers[epoch]

        train_sampler.set_epoch(epoch)

        loss_meter     = AverageMeter()
        inv_loss_meter = AverageMeter()
        sig_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()

        for itr, (udata, view_masks, _masks_attention) in enumerate(train_loader):

            global_step = epoch * ipe + itr

            # Move data to device
            for key, val in udata.items():
                if isinstance(val, torch.Tensor):
                    udata[key] = val.to(device, non_blocking=True)
            # view_masks: list of V tensors [B, view_len]
            view_masks = [m.to(device, non_blocking=True) for m in view_masks]

            compute_grad_stats = (WORLD_RANK == 0) and (itr % log_freq == 0)

            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()

            # Update adversarial lambda schedule
            if use_adversarial:
                lambd = adversarial_lambda_schedule(
                    global_step, total_steps, adv_schedule_gamma)
                for ddp_head in ddp_adv_heads.values():
                    ddp_head.module.set_lambd(lambd)

            # --------------------------------------------------------------
            # Forward pass
            # --------------------------------------------------------------
            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                # Encode all V views in one call via EncoderMultiMaskWrapper.
                # view_embeds: list of V tensors, each [B, view_len, D]
                view_embeds, _ = encoder(
                    batch=udata,
                    masks=view_masks,
                    masks_attention=None,
                )

                # Pool each view to [B, D] and project to [B, proj_dim]
                projs: list[torch.Tensor] = []
                pooled_views: list[torch.Tensor] = []
                for x_v in view_embeds:
                    emb_v = _pool(x_v, pooling)  # [B, D]
                    pooled_views.append(emb_v)
                    projs.append(proj_head(emb_v))  # [B, proj_dim]

                # Stack: [V, B, proj_dim]
                proj = torch.stack(projs, dim=0)

                # -- Invariance loss --
                inv_loss = (proj.mean(0, keepdim=True) - proj).square().mean()

                # -- SIGReg loss --
                sigreg_loss = sigreg(proj)

                # -- Adversarial loss --
                adv_loss = torch.tensor(0.0, device=device)
                adv_acc = torch.tensor(0.0, device=device)
                if use_adversarial:
                    for label_name, ddp_head in ddp_adv_heads.items():
                        raw_labels = udata[label_name].squeeze(-1)
                        remapped = torch.zeros_like(raw_labels, dtype=torch.long)
                        for src, tgt in adv_label_map[label_name].items():
                            remapped[raw_labels == src] = tgt

                        for emb_v in pooled_views:
                            logits = ddp_head(emb_v)
                            #if itr < 10: # DEBUG
                            #    print("logits:", logits[:4])
                            #    print("remapped:", remapped[:4])
                            #    print("lambd:", lambd)
                            adv_loss = adv_loss + F.cross_entropy(logits, remapped, label_smoothing=0.1)
                            adv_acc = adv_acc + (logits.argmax(dim=-1) == remapped).float().mean()

                    adv_loss = adv_loss / (V * len(adversarial_labels))
                    adv_acc = adv_acc / (V * len(adversarial_labels))

                # -- Combined ST-JEPA loss --
                loss = ((1.0 - lamb) * inv_loss
                        + lamb * sigreg_loss
                        + adv_weight * adv_loss)

            # --------------------------------------------------------------
            # Backward pass
            # --------------------------------------------------------------
            _enc_norm, _proj_norm = 0.0, 0.0
            loss.backward()

            if use_adversarial and (global_step % adv_disc_update_freq != 0):
                for ddp_head in ddp_adv_heads.values():
                    ddp_head.zero_grad(set_to_none=True)

            if (epoch >= warmup) and (clip_grad is not None):
                _enc_norm = torch.nn.utils.clip_grad_norm_(
                    encoder.parameters(), clip_grad)
                _proj_norm = torch.nn.utils.clip_grad_norm_(
                    proj_head.parameters(), clip_grad)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # --------------------------------------------------------------
            # Logging
            # --------------------------------------------------------------
            if compute_grad_stats:
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
            else:
                grad_stats = None

            loss_meter.update(float(loss))
            inv_loss_meter.update(float(inv_loss))
            sig_loss_meter.update(float(sigreg_loss))
            if use_adversarial:
                adv_loss_meter.update(float(adv_loss))

            if WORLD_RANK == 0 and (itr % log_freq == 0):
                log_dict = {
                    "loss/total":    float(loss),
                    "loss/inv":      float(inv_loss),
                    "loss/sigreg":   float(sigreg_loss),
                    "lr":            float(_new_lr),
                    "wd":            float(_new_wd),
                    "epoch":         int(epoch),
                    "global_norm_enc": float(grad_stats.global_norm)
                    if grad_stats is not None else 0.0,
                }
                if use_adversarial:
                    log_dict["loss/adversarial"] = float(adv_loss)
                    log_dict["adv_lambda"] = lambd
                    log_dict["adv_acc"] = float(adv_acc)
                    log_dict["adv_disc_update"] = int(global_step % adv_disc_update_freq == 0)

                wandb.log(log_dict)

            assert not np.isnan(float(loss)), "Loss is NaN — aborting."

            if itr % checkpoint_freq_iter == 0:
                logger.info(
                    f"Saving checkpoint at epoch {epoch}, iter {itr}.")
                save_checkpoint(epoch, itr // checkpoint_freq_iter)

        # -- End of epoch --
        adv_str = f" | adv {adv_loss_meter.avg:.4f}" if use_adversarial else ""
        logger.info(
            f"Epoch {epoch} | loss {loss_meter.avg:.4f} | "
            f"inv {inv_loss_meter.avg:.4f} | "
            f"sigreg {sig_loss_meter.avg:.4f}{adv_str}")
        save_checkpoint(epoch)
