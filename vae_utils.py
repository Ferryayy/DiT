from pathlib import Path

from diffusers.models import AutoencoderKL


VAE_REPOS = {
    "ema": "stabilityai/sd-vae-ft-ema",
    "mse": "stabilityai/sd-vae-ft-mse",
}


def _resolve_local_vae_path(vae_path):
    path = Path(vae_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Could not find local VAE directory at {path}")
    if not path.is_dir():
        raise ValueError(f"vae_path must point to a diffusers model directory, but got a file: {path}")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Local VAE directory {path} is missing config.json. "
            f"Please point vae_path to a diffusers-compatible model directory."
        )

    weight_patterns = (
        "diffusion_pytorch_model*.safetensors",
        "diffusion_pytorch_model*.bin",
    )
    has_weights = any(any(path.glob(pattern)) for pattern in weight_patterns)
    if not has_weights:
        raise FileNotFoundError(
            f"Local VAE directory {path} does not contain diffusion_pytorch_model weights. "
            f"Please point vae_path to a diffusers-compatible model directory."
        )
    return str(path.resolve())


def load_vae(vae, vae_path=None, device=None):
    if vae_path:
        source = _resolve_local_vae_path(vae_path)
        loaded_vae = AutoencoderKL.from_pretrained(source, local_files_only=True)
        source_kind = "local"
    else:
        if vae not in VAE_REPOS:
            raise ValueError(f"Unsupported VAE variant: {vae}. Expected one of {list(VAE_REPOS)}.")
        source = VAE_REPOS[vae]
        loaded_vae = AutoencoderKL.from_pretrained(source)
        source_kind = "hub"

    if device is not None:
        loaded_vae = loaded_vae.to(device)
    return loaded_vae, source, source_kind
