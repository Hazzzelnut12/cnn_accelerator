"""Validate that the project environment can run CNN training on CUDA."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch import nn
import torchvision


def check_environment() -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run with the project's Python 3.12 environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and PyTorch build.")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    left, right = torch.randn(64, 64), torch.randn(64, 64)
    cpu_result = left @ right
    gpu_result = (left.cuda() @ right.cuda()).cpu()
    torch.testing.assert_close(gpu_result, cpu_result, atol=1e-4, rtol=1e-4)

    # Exercise the planned layer shapes, gradient calculation, and parameter updates.
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10),
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.rand(16, 1, 28, 28, device="cuda")
    labels = torch.arange(16, device="cuda") % 10
    before = model[0].weight.detach().clone()
    logits = model(images)
    if logits.shape != (16, 10):
        raise RuntimeError(f"Unexpected CNN output shape: {tuple(logits.shape)}")
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError("CNN loss is not finite.")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
    optimizer.step()
    torch.cuda.synchronize()
    if torch.equal(before, model[0].weight):
        raise RuntimeError("The optimizer did not update the convolution weights.")

    return {
        "status": "PASS",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "matmul_max_abs_error": (gpu_result - cpu_result).abs().max().item(),
        "cnn_output_shape": list(logits.shape),
        "cnn_loss": loss.item(),
        "checks": ["cpu_cuda_matmul_agreement", "cnn_forward_backward", "optimizer_update"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the validation result as JSON.")
    args = parser.parse_args()
    try:
        report = check_environment()
    except Exception as error:
        report = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    result = json.dumps(report, indent=2, ensure_ascii=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
