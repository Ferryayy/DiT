"""
Download Stability AI VAE snapshots for local offline loading.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


VAE_REPOS = {
    "ema": "stabilityai/sd-vae-ft-ema",
    "mse": "stabilityai/sd-vae-ft-mse",
}


ALLOWED_PATTERNS = [
    "config.json",
    "diffusion_pytorch_model.safetensors",
]


def download_vae(variant, output_dir, revision):
    repo_id = VAE_REPOS[variant]
    local_dir = Path(output_dir) / repo_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=ALLOWED_PATTERNS,
    )
    print(f"Downloaded {repo_id}@{revision} to {local_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["ema", "mse", "both"], default="ema")
    parser.add_argument("--output-dir", type=str, default="pretrained_models/vae")
    parser.add_argument("--revision", type=str, default="main")
    args = parser.parse_args()

    variants = list(VAE_REPOS) if args.variant == "both" else [args.variant]
    for variant in variants:
        download_vae(variant, args.output_dir, args.revision)

    print("snapshot_download only fetches the selected repository revision, not git history.")


if __name__ == "__main__":
    main()
