"""저장된 float 모델로 MNIST 공식 test 10,000장을 평가하고 기록된 결과와 비교."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from train import MnistCNN, validate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "mnist_float_v1",
                        help="best.pt와 results.json이 있는 폴더")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_dir / "best.pt", map_location=device, weights_only=True)
    model = MnistCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = MNIST(root=PROJECT / "data", train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    loss, accuracy = validate(model, test_loader, nn.CrossEntropyLoss(), device, log_interval=10**9, phase="test")
    print(f"모델: {args.model_dir} (epoch {checkpoint['epoch']}) | 장치: {device}")
    print(f"test loss {loss:.4f} | accuracy {100 * accuracy:.2f}% ({round(accuracy * len(test_dataset))}/{len(test_dataset)})")

    # 학습 때 기록한 결과와 비교. 정답 개수가 같으면 가중치·환경이 제대로 공유된 것.
    recorded = json.loads((args.model_dir / "results.json").read_text(encoding="utf-8"))
    same = round(accuracy * len(test_dataset)) == round(recorded["test_accuracy"] * len(test_dataset))
    print(f"기록된 test accuracy {100 * recorded['test_accuracy']:.2f}% → {'일치' if same else '불일치'}")
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
