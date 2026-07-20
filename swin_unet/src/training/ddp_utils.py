import os
import builtins
import torch
import torch.distributed as dist

def setup_for_distributed(is_master: bool) -> None:
    """
    Overrides print to only output on the master process (rank 0).
    """
    builtin_print = builtins.print

    def new_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = new_print

def init_distributed_mode() -> dict:
    """
    Initializes Distributed Data Parallel (DDP) mode if variables are set by torchrun/dist.
    Returns a dict with ddp status information.
    """
    info = {
        "distributed": False,
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        info["rank"] = int(os.environ["RANK"])
        info["world_size"] = int(os.environ["WORLD_SIZE"])
        info["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))
        info["distributed"] = True
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        info["rank"] = int(os.environ["OMPI_COMM_WORLD_RANK"])
        info["world_size"] = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        info["local_rank"] = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        info["distributed"] = True

    if info["distributed"]:
        torch.cuda.set_device(info["local_rank"])
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=info["world_size"],
            rank=info["rank"]
        )
        info["device"] = f"cuda:{info['local_rank']}"
        dist.barrier()
        setup_for_distributed(info["rank"] == 0)
        print(f"[ddp] initialized process rank {info['rank']} (world_size {info['world_size']})")
    else:
        setup_for_distributed(True)

    return info

def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
