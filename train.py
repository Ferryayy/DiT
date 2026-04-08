# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import yaml

from models import DiT_models
from diffusion import create_diffusion
from vae_utils import load_vae


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def parse_gpu_ids(gpu_ids):
    if gpu_ids is None:
        return None
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    if isinstance(gpu_ids, str):
        gpu_ids = gpu_ids.strip()
        if not gpu_ids:
            return None
        return [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
    if isinstance(gpu_ids, (list, tuple)):
        return [int(x) for x in gpu_ids]
    raise ValueError("gpu_ids must be an int, a comma-separated string, or a list of ints.")


def setup_distributed(args):
    configured_mode = args.train_mode.lower()
    launched_with_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    world_size_from_env = int(os.environ.get("WORLD_SIZE", "1"))

    if configured_mode not in {"single", "ddp", "auto"}:
        raise ValueError("train_mode must be one of: single, ddp, auto.")
    if configured_mode == "ddp" and not launched_with_torchrun:
        raise ValueError("train_mode=ddp requires torchrun. Please use start.sh or torchrun to launch training.")
    if configured_mode == "ddp" and len(args.gpu_ids) < 2:
        raise ValueError("train_mode=ddp requires at least 2 GPU ids.")
    if configured_mode == "single" and world_size_from_env > 1:
        raise ValueError("train_mode=single cannot be launched with multi-process torchrun.")

    distributed = launched_with_torchrun and world_size_from_env > 1
    if configured_mode == "auto":
        distributed = launched_with_torchrun and world_size_from_env > 1

    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    return distributed, rank, world_size, local_rank, device


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size."
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, local_rank={local_rank}, seed={seed}, world_size={world_size}, distributed={distributed}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae, vae_source, vae_source_kind = load_vae(args.vae, args.vae_path, device)
    logger.info(f"Loaded VAE from {'local path' if vae_source_kind == 'local' else 'Hugging Face'}: {vae_source}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    ) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    model_without_ddp = model.module if distributed else model
    update_ema(ema, model_without_ddp, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model_without_ddp)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if distributed:
                    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


def load_args(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    defaults = {
        "data_path": None,
        "results_dir": "results",
        "model": "DiT-XL/2",
        "image_size": 256,
        "num_classes": 1000,
        "epochs": 1400,
        "global_batch_size": 256,
        "global_seed": 0,
        "vae": "ema",
        "vae_path": None,
        "num_workers": 4,
        "log_every": 100,
        "ckpt_every": 50_000,
        "train_mode": "single",
        "gpu_ids": "0",
        "master_addr": "127.0.0.1",
        "master_port": 29500,
    }
    defaults.update(config)

    assert defaults["data_path"], "Please set data_path in the config file."
    assert defaults["model"] in DiT_models, f"Model must be one of {list(DiT_models.keys())}."
    assert defaults["image_size"] in [256, 512], "Image size must be 256 or 512."
    assert defaults["vae"] in ["ema", "mse"], "VAE must be ema or mse."
    defaults["gpu_ids"] = parse_gpu_ids(defaults["gpu_ids"])
    assert defaults["train_mode"] in ["single", "ddp", "auto"], "train_mode must be single, ddp, or auto."
    assert defaults["gpu_ids"], "Please provide at least one GPU id in gpu_ids."
    if defaults["train_mode"] == "single":
        assert len(defaults["gpu_ids"]) == 1, "single mode requires exactly one GPU id."
    if defaults["train_mode"] == "ddp":
        assert len(defaults["gpu_ids"]) >= 2, "ddp mode requires at least two GPU ids."

    return argparse.Namespace(**defaults)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Optional local diffusers VAE directory. If set, this overrides vae_path in the config.")
    cli_args = parser.parse_args()
    args = load_args(cli_args.config)
    if cli_args.vae_path is not None:
        args.vae_path = cli_args.vae_path
    main(args)
