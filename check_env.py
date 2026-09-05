"""Print the runtime prerequisites for the SAR segmentation pipeline."""

import importlib.util
import platform

import torch


def main() -> None:
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Total VRAM: {props.total_memory / (1024 ** 3):.2f} GB")

    missing = []
    for package in ("segmentation_models_pytorch", "albumentations"):
        available = importlib.util.find_spec(package) is not None
        print(f"{package} importable: {available}")
        if not available:
            missing.append(package)

    if missing:
        print("Install missing packages with:")
        print("pip install segmentation-models-pytorch albumentations")
        print("If PyTorch itself needs installation, choose a CUDA-matched command from https://pytorch.org/get-started/locally/.")


if __name__ == "__main__":
    main()
