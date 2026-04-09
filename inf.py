import argparse
from datetime import datetime
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from vae_utils import load_vae


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def rank0_print(message):
    if get_rank() == 0:
        print(message, flush=True)


def parse_gpu_ids(gpu_ids):
    if gpu_ids is None:
        return None
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    if isinstance(gpu_ids, str):
        gpu_ids = gpu_ids.strip()
        if not gpu_ids or gpu_ids.lower() == "auto":
            return None
        return [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
    if isinstance(gpu_ids, (list, tuple)):
        return [int(x) for x in gpu_ids]
    raise ValueError("gpu_ids must be an int, a comma-separated string, or a list of ints.")


def parse_class_labels(class_labels):
    if class_labels is None:
        return None
    if isinstance(class_labels, int):
        return [class_labels]
    if isinstance(class_labels, str):
        class_labels = class_labels.strip()
        if not class_labels:
            return None
        return [int(x.strip()) for x in class_labels.split(",") if x.strip()]
    if isinstance(class_labels, (list, tuple)):
        if not class_labels:
            return None
        return [int(x) for x in class_labels]
    raise ValueError("class_labels must be an int, a comma-separated string, or a list of ints.")


def setup_distributed(args):
    configured_mode = args.infer_mode.lower()
    launched_with_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    world_size_from_env = int(os.environ.get("WORLD_SIZE", "1"))

    if configured_mode not in {"single", "ddp", "auto"}:
        raise ValueError("infer_mode must be one of: single, ddp, auto.")
    if configured_mode == "ddp" and not launched_with_torchrun:
        raise ValueError("infer_mode=ddp requires torchrun. Please use ref.sh or torchrun to launch inference.")
    if configured_mode == "single" and world_size_from_env > 1:
        raise ValueError("infer_mode=single cannot be launched with multi-process torchrun.")
    if configured_mode == "ddp" and torch.cuda.is_available() and args.gpu_ids is not None and len(args.gpu_ids) < 2:
        raise ValueError("infer_mode=ddp requires at least 2 GPU ids.")

    distributed = launched_with_torchrun and world_size_from_env > 1
    if configured_mode == "auto":
        distributed = launched_with_torchrun and world_size_from_env > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            local_rank = 0
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

    return distributed, rank, world_size, local_rank, device


def build_global_labels(args):
    if args.class_labels is not None:
        return args.class_labels

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.global_seed)
    labels = torch.randint(0, args.num_classes, (args.num_samples,), generator=generator)
    return labels.tolist()


def resolve_run_name(args, total_samples, ckpt_name, vae_name):
    if args.run_name:
        return args.run_name
    sample_tag = f"labels-{total_samples}" if args.class_labels is not None else f"random-{total_samples}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{args.model.replace('/', '-')}-{ckpt_name}-size-{args.image_size}-vae-{vae_name}-"
        f"{sample_tag}-cfg-{args.cfg_scale}-steps-{args.num_sampling_steps}-seed-{args.global_seed}-{timestamp}"
    )


def ensure_output_dirs(run_dir):
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def save_resolved_config(args, run_dir, total_samples, image_dir, vae_source, vae_source_kind, world_size):
    payload = dict(vars(args))
    payload["gpu_ids"] = args.gpu_ids if args.gpu_ids is None else list(args.gpu_ids)
    payload["class_labels"] = args.class_labels if args.class_labels is None else list(args.class_labels)
    payload["resolved_total_samples"] = int(total_samples)
    payload["resolved_world_size"] = int(world_size)
    payload["resolved_cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    payload["resolved_output_dir"] = str(run_dir)
    payload["resolved_image_dir"] = str(image_dir)
    payload["resolved_vae_source"] = vae_source
    payload["resolved_vae_source_kind"] = vae_source_kind
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def build_grid_from_sample_dir(image_dir, grid_path, nrow, max_images):
    image_paths = sorted(image_dir.glob("*.png"))[:max_images]
    if not image_paths:
        return

    tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    save_image(torch.stack(tensors), grid_path, nrow=nrow)


def create_npz_from_sample_dir(image_dir, npz_path, num_expected):
    image_paths = sorted(image_dir.glob("*.png"))[:num_expected]
    samples = []
    for image_path in tqdm(image_paths, desc="Building .npz file from samples"):
        sample_pil = Image.open(image_path).convert("RGB")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")


def validate_args(args):
    assert args.model in DiT_models, f"Model must be one of {list(DiT_models.keys())}."
    assert args.image_size in [256, 512], "image_size must be 256 or 512."
    assert args.image_size % 8 == 0, "image_size must be divisible by 8."
    assert args.num_classes > 0, "num_classes must be positive."
    assert args.per_proc_batch_size > 0, "per_proc_batch_size must be positive."
    assert args.num_sampling_steps > 0, "num_sampling_steps must be positive."
    assert args.cfg_scale >= 1.0, "cfg_scale should be >= 1.0."
    assert args.vae in ["ema", "mse"], "vae must be ema or mse."
    assert args.grid_nrow > 0, "grid_nrow must be positive."
    assert args.grid_max_images > 0, "grid_max_images must be positive."
    assert args.output_dir, "output_dir must not be empty."

    if args.class_labels is not None:
        assert len(args.class_labels) > 0, "class_labels must not be empty."
        for label in args.class_labels:
            assert 0 <= label < args.num_classes, f"class label {label} is out of range [0, {args.num_classes})."
    else:
        assert args.num_samples is not None and args.num_samples > 0, "num_samples must be set when class_labels is empty."

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000


def load_args(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    defaults = {
        "infer_mode": "auto",
        "gpu_ids": "auto",
        "master_addr": "127.0.0.1",
        "master_port": 29501,
        "model": "DiT-XL/2",
        "ckpt": None,
        "image_size": 256,
        "num_classes": 1000,
        "vae": "mse",
        "vae_path": None,
        "cfg_scale": 4.0,
        "num_sampling_steps": 250,
        "per_proc_batch_size": 8,
        "global_seed": 0,
        "tf32": True,
        "output_dir": "inference_outputs",
        "run_name": None,
        "class_labels": [207, 360, 387, 974, 88, 979, 417, 279],
        "num_samples": None,
        "save_grid": True,
        "grid_nrow": 4,
        "grid_max_images": 64,
        "save_npz": False,
    }
    defaults.update(config)

    defaults["gpu_ids"] = parse_gpu_ids(defaults["gpu_ids"])
    defaults["class_labels"] = parse_class_labels(defaults["class_labels"])

    if defaults["infer_mode"] == "single" and defaults["gpu_ids"] is not None:
        assert len(defaults["gpu_ids"]) <= 1, "single mode supports at most one explicit GPU id."
    if defaults["infer_mode"] == "ddp" and defaults["gpu_ids"] is not None:
        assert len(defaults["gpu_ids"]) >= 2, "ddp mode requires at least two explicit GPU ids."

    args = argparse.Namespace(**defaults)
    validate_args(args)
    return args


def main(args):
    validate_args(args)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    seed = args.global_seed * max(world_size, 1) + rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rank0_print(
        f"Starting inference with rank={rank}, local_rank={local_rank}, seed={seed}, "
        f"world_size={world_size}, device={device}."
    )

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae, vae_source, vae_source_kind = load_vae(args.vae, args.vae_path, device)
    rank0_print(f"Loaded VAE from {'local path' if vae_source_kind == 'local' else 'Hugging Face'}: {vae_source}")

    global_labels = build_global_labels(args)
    total_samples = len(global_labels)
    using_cfg = args.cfg_scale > 1.0
    label_width = max(4, len(str(args.num_classes - 1)))
    ckpt_name = Path(args.ckpt).stem if args.ckpt else "pretrained"
    vae_name = f"local-{Path(vae_source).name}" if vae_source_kind == "local" else args.vae

    if distributed:
        run_name_holder = [resolve_run_name(args, total_samples, ckpt_name, vae_name) if rank == 0 else None]
        dist.broadcast_object_list(run_name_holder, src=0)
        run_name = run_name_holder[0]
    else:
        run_name = resolve_run_name(args, total_samples, ckpt_name, vae_name)

    run_dir = Path(args.output_dir) / run_name
    image_dir = ensure_output_dirs(run_dir)
    if rank == 0:
        save_resolved_config(args, run_dir, total_samples, image_dir, vae_source, vae_source_kind, world_size)
        print(f"Saving samples to {image_dir}", flush=True)
    if distributed:
        dist.barrier()

    local_indices = list(range(rank, total_samples, world_size))
    batch_starts = range(0, len(local_indices), args.per_proc_batch_size)
    progress = tqdm(batch_starts, desc="Sampling", disable=rank != 0)

    for batch_start in progress:
        batch_indices = local_indices[batch_start: batch_start + args.per_proc_batch_size]
        if not batch_indices:
            continue

        batch_labels = [global_labels[index] for index in batch_indices]
        batch_size = len(batch_indices)
        z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
        y = torch.tensor(batch_labels, device=device, dtype=torch.long)

        if using_cfg:
            z = torch.cat([z, z], dim=0)
            y_null = torch.full((batch_size,), args.num_classes, device=device, dtype=torch.long)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for sample, global_index, label in zip(samples, batch_indices, batch_labels):
            filename = f"{global_index:06d}_cls{label:0{label_width}d}.png"
            Image.fromarray(sample).save(image_dir / filename)

    if distributed:
        dist.barrier()
    if rank == 0:
        if args.save_grid:
            build_grid_from_sample_dir(
                image_dir=image_dir,
                grid_path=run_dir / "grid.png",
                nrow=args.grid_nrow,
                max_images=min(args.grid_max_images, total_samples),
            )
        if args.save_npz:
            create_npz_from_sample_dir(image_dir=image_dir, npz_path=run_dir / "samples.npz", num_expected=total_samples)
        print("Inference done.", flush=True)
    if distributed:
        dist.barrier()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Optional local diffusers VAE directory. If set, this overrides vae_path in the config.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional DiT checkpoint path. If set, this overrides ckpt in the config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Optional output directory. If set, this overrides output_dir in the config.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional run name. If set, this overrides run_name in the config.")
    parser.add_argument("--class-labels", type=str, default=None,
                        help="Optional comma-separated class labels. If set, this overrides class_labels in the config.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional number of random samples. Used when class_labels is not provided.")
    cli_args = parser.parse_args()

    args = load_args(cli_args.config)
    if cli_args.vae_path is not None:
        args.vae_path = cli_args.vae_path
    if cli_args.ckpt is not None:
        args.ckpt = cli_args.ckpt
    if cli_args.output_dir is not None:
        args.output_dir = cli_args.output_dir
    if cli_args.run_name is not None:
        args.run_name = cli_args.run_name
    if cli_args.class_labels is not None:
        args.class_labels = parse_class_labels(cli_args.class_labels)
    if cli_args.num_samples is not None:
        args.num_samples = cli_args.num_samples

    validate_args(args)
    main(args)
