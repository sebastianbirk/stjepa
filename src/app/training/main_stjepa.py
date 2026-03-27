"""
Entry point for distributed ST-JEPA pre-training on spatial transcriptomics data.

Launch with torchrun (single/multi-node) or mpirun; see the shell scripts in
this directory for example commands.

Example (single node, 4 GPUs):

    torchrun --standalone --nproc_per_node=4 \\
        src/app/training/main_stjepa.py \\
        --fname configs/model/example/stjepa.yaml

Example (SLURM multi-node, see train_model_slurm_torchrun_*.sh):

    MASTER_ADDR=... MASTER_PORT=... torchrun ... main_stjepa.py --fname ...
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from socket import gethostname

import torch
import torch.distributed as dist

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from app.training.train_stjepa import train_stjepa
from stjepa.datasets.utils import prepare_dataset
from stjepa.utils.config import create_params_from_YAML_wandb_config
import wandb

warnings.filterwarnings("ignore")

logging.basicConfig()
logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Distributed info helpers (identical to main.py)
# ---------------------------------------------------------------------------

def get_distributed_info():
    if "SLURM_PROCID" in os.environ:
        WORLD_RANK = int(os.environ["SLURM_PROCID"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert GPUS_PER_NODE == torch.cuda.device_count()
        print(
            f"Hello from rank {WORLD_RANK} of {WORLD_SIZE} on "
            f"{gethostname()} where there are {GPUS_PER_NODE} allocated "
            f"GPUs per node.",
            flush=True,
        )
        LOCAL_RANK = WORLD_RANK - GPUS_PER_NODE * (WORLD_RANK // GPUS_PER_NODE)
    if "LOCAL_RANK" in os.environ:
        LOCAL_RANK  = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE  = int(os.environ["WORLD_SIZE"])
        WORLD_RANK  = int(os.environ["RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        LOCAL_RANK  = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        WORLD_SIZE  = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        WORLD_RANK  = int(os.environ["OMPI_COMM_WORLD_RANK"])
    else:
        sys.exit("Cannot find distributed environment variables.")

    print(
        f"World rank: {WORLD_RANK}, Local rank: {LOCAL_RANK}, "
        f"World size: {WORLD_SIZE}"
    )
    return WORLD_RANK, LOCAL_RANK, WORLD_SIZE


def setup_for_distributed(is_master: bool):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    WORLD_RANK, LOCAL_RANK, WORLD_SIZE = get_distributed_info()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Backend for distributed training.",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default="configs/model/example/stjepa.yaml",
        help="Path to the ST-JEPA config YAML file.",
    )
    args = parser.parse_args()

    experiment_name = os.environ.get("EXPERIMENT_NAME", "stjepa")
    run_name        = os.environ.get("RUN_NAME", "run")
    run_id          = f"{experiment_name}_{run_name}"

    params = create_params_from_YAML_wandb_config(args.fname, logger)
    logger.info(f"Called with params from {args.fname}.")

    if WORLD_RANK == 0:
        wandb.init(
            settings=wandb.Settings(init_timeout=120),
            project="stjepa-pretraining",
            id=run_id,
            resume="allow",
            group="multi_node_training",
            mode="online",
        )
        current_timestamp = (
            datetime.now().strftime("%d%m%Y_%H%M%S")
            + f"_{datetime.now().microsecond // 1000:03d}"
        )
        print(f"Run timestamp: {current_timestamp}.")
        if not wandb.run.resumed:
            wandb.config.run_timestamp = current_timestamp

        artifact_root = "../stjepa-reproducibility/artifacts"
        folder_path = (
            params["state"]["folder_path"]
            or os.path.join(
                artifact_root,
                params["data"]["dataset_name"],
                current_timestamp,
            )
        )
    else:
        folder_path = None

    print(
        f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )

    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(
        backend=args.backend,
        init_method=(
            f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        ),
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
    )

    dist.barrier()
    setup_for_distributed(LOCAL_RANK == 0)

    train_dataset, _, _ = prepare_dataset(params, train_mode=True)

    train_stjepa(
        params,
        train_dataset,
        save_folder_path=folder_path,
        LOCAL_RANK=LOCAL_RANK,
        WORLD_RANK=WORLD_RANK,
    )


if __name__ == "__main__":
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    try:
        nccl_version = torch.cuda.nccl.version()
    except AttributeError:
        nccl_version = "NCCL not available."
    print(f"torch.cuda.nccl.version(): {nccl_version}")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())

    os.environ["TORCH_CPP_LOG_LEVEL"]        = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"]    = "DETAIL"

    main()
