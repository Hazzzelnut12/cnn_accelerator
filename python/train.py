import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import torch
from torch import Generator, Tensor, nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomAffine, ToTensor

# 프로젝트 루트(python/의 상위). 데이터는 data/, 학습 실행 결과는 build/training/에 저장.
PROJECT = Path(__file__).resolve().parent.parent


# 학습용 증강: 작은 회전(±10도)·이동(±10%)·크기(90~110%). 빈 영역은 0(검은 배경)으로 채움.
# 숫자 의미가 바뀌는 좌우 반전·큰 회전은 사용하지 않음.
AUGMENT_TRANSFORM = Compose([
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ToTensor(),
])


# 블록 1: MNIST용 CNN 구조 정의
# 텐서 차원: (N, C, H, W) = (이미지 수, 채널 수, 높이, 너비)
class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 특징 추출: 합성곱 → ReLU → 크기 축소
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 숫자 분류: 특징을 펼친 뒤 0~9의 점수 계산
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: Tensor) -> Tensor:
        # (N, 1, 28, 28) → (N, 8, 28, 28) → (N, 8, 14, 14)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # (N, 8, 14, 14) → (N, 16, 14, 14) → (N, 16, 7, 7)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # (N, 16, 7, 7) → (N, 784) → (N, 32) → (N, 10)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        # Softmax 전의 점수(logits). 아직 확률은 아님.
        return x


# 블록 2: MNIST 불러오기와 학습용/검증용 분할
def load_mnist(seed: int = 42, augment: bool = False) -> tuple[Subset, Subset]:
    # 실행 위치와 관계없이 프로젝트 루트의 data 폴더에 저장.
    data_dir = PROJECT / "data"

    # train=True: 공식 학습 데이터 60,000장 사용.
    # ToTensor: 픽셀 0~255 → float32 0~1, 크기 (1, 28, 28).
    dataset = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # 같은 seed를 사용하면 매번 같은 이미지들이 같은 그룹에 배정됨.
    # 두 그룹은 겹치지 않으며, 공식 test 10,000장은 분할에 사용하지 않음.
    train_dataset, val_dataset = random_split(
        dataset,
        lengths=[55_000, 5_000],
        generator=Generator().manual_seed(seed),
    )

    # 증강 시 학습용만 증강 transform으로 다시 읽음. 같은 index를 사용하므로 분할은 동일하고,
    # 검증용은 원본 그대로 평가.
    if augment:
        augmented = MNIST(root=data_dir, train=True, download=True, transform=AUGMENT_TRANSFORM)
        train_dataset = Subset(augmented, train_dataset.indices)
    return train_dataset, val_dataset


# 블록 3: 데이터를 batch 단위로 공급하는 DataLoader
def create_data_loaders(
    train_dataset: Subset,
    val_dataset: Subset,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    # 학습용: 한 바퀴(epoch) 읽을 때마다 이미지 순서를 새로 섞음.
    # seed는 실행 간 섞이는 순서를 재현하기 위한 값. 분할 자체는 바뀌지 않음.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=Generator().manual_seed(seed),
        num_workers=0,  # 별도 작업 프로세스 없이 현재 프로세스에서 로딩.
        drop_last=False,  # 마지막 batch가 64장보다 작아도 모두 사용.
    )

    # 검증용: 같은 순서로 읽고, 마지막 이미지까지 빠짐없이 평가.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


# 블록 4: 학습 장치, 모델, 손실 함수, 최적화기 준비
def prepare_training(
    learning_rate: float = 0.001,
    seed: int = 42,
    weight_decay: float = 0.0,
) -> tuple[torch.device, MnistCNN, nn.CrossEntropyLoss, torch.optim.Optimizer]:
    # 모델의 초기 가중치를 같은 조건에서 재현하기 위한 seed.
    torch.manual_seed(seed)

    # CUDA를 사용할 수 있으면 GPU, 그렇지 않으면 CPU 선택.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistCNN().to(device)

    # 모델의 logits와 정답 숫자(0~9)를 받아 손실 계산.
    # CrossEntropyLoss에 넣기 전에 Softmax를 따로 적용하지 않음.
    criterion = nn.CrossEntropyLoss()

    # 학습 시 모델 가중치를 갱신할 도구. 생성만으로는 갱신되지 않음.
    # lr은 갱신 크기를 조절하는 학습률이며, 0.001을 초기 설정으로 사용.
    # weight decay를 쓰면 손실과 분리해 가중치를 줄이는 AdamW 사용. 0이면 기준 실험과 같은 Adam.
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return device, model, criterion, optimizer


# 블록 5: 학습 데이터를 한 바퀴(epoch) 사용해 가중치 갱신
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 100,
) -> tuple[float, float]:
    """기본 CrossEntropyLoss 사용. 이미지당 평균 손실과 정확도(0~1)를 반환."""
    model.train()  # 학습 모드. 실제 가중치 갱신은 아래 optimizer.step()에서 수행.
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)  # 이전 batch의 기울기 초기화.
        logits = model(images)  # 순전파: 이미지에서 숫자별 점수 계산.
        loss = criterion(logits, labels)  # 점수와 정답을 비교해 손실 계산.
        loss.backward()  # 역전파: 각 가중치의 기울기 계산.
        optimizer.step()  # 기울기를 이용해 가중치 갱신.

        # loss는 batch 평균이므로 이미지 수를 곱해 전체 손실을 누적.
        # 마지막 작은 batch도 이미지 수에 맞는 비중으로 반영.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(train_loader):
            logging.info(
                "  학습 %d/%d batch | loss %.4f | accuracy %.2f%%",
                batch_index, len(train_loader),
                total_loss / total_samples, 100 * total_correct / total_samples,
            )

    if total_samples == 0:
        raise ValueError("학습 데이터가 비어 있습니다.")

    # 학습 지표는 각 batch를 처리한 시점의 예측으로 집계한 값.
    return total_loss / total_samples, total_correct / total_samples


# 블록 6: 가중치를 수정하지 않고 검증 데이터의 손실·정확도 측정
@torch.inference_mode()  # 검증에서는 역전파에 필요한 계산 기록을 만들지 않음.
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    log_interval: int = 100,
    phase: str = "검증",
) -> tuple[float, float]:
    """기본 CrossEntropyLoss 사용. 이미지당 평균 손실과 정확도(0~1)를 반환."""
    model.eval()  # 평가 모드. 다음 학습 호출은 train_one_epoch에서 모드를 복원.
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(val_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(val_loader):
            logging.info(
                "  %s %d/%d batch | loss %.4f | accuracy %.2f%%",
                phase, batch_index, len(val_loader),
                total_loss / total_samples, 100 * total_correct / total_samples,
            )

    if total_samples == 0:
        raise ValueError("검증 데이터가 비어 있습니다.")

    return total_loss / total_samples, total_correct / total_samples


# 블록 7: 실행 설정. --help로 옵션을 확인할 수 있음.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST CNN 학습·검증·최고 모델 저장")
    parser.add_argument("--epochs", type=int, default=20, help="학습 횟수 (기본: 20)")
    parser.add_argument("--batch-size", type=int, default=64, help="batch 크기 (기본: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam 학습률 (기본: 0.001)")
    parser.add_argument("--augment", action="store_true", help="학습용 데이터에 작은 회전·이동·크기 증강 적용")
    parser.add_argument(
        "--scheduler", choices=["none", "cosine"], default="none",
        help="epoch별 학습률 조정. cosine: 마지막 epoch까지 0에 가깝게 감소 (기본: none)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="0보다 크면 AdamW로 가중치 감쇠 적용 (기본: 0, Adam)",
    )
    parser.add_argument("--seed", type=int, default=42, help="분할·초기화 seed (기본: 42)")
    parser.add_argument("--log-interval", type=int, default=100, help="몇 batch마다 진행 상황을 표시할지")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT / "build" / "training",
        help="실행별 결과 폴더를 만들 위치",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.log_interval <= 0:
        parser.error("epochs, batch-size, log-interval은 양수여야 합니다.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr은 유한한 양수여야 합니다.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay는 0 이상의 유한한 값이어야 합니다.")
    if not 0 <= args.seed < 2**63:
        parser.error("seed는 0 이상 2**63 미만이어야 합니다.")
    return args


# 블록 8: 여러 epoch 실행, 경과 기록, 최고 모델 저장, 최종 test 평가
def main() -> None:
    args = parse_args()
    # 매 실행마다 별도 폴더를 만들어 이전 가중치와 기록을 보존.
    started_at = datetime.now(timezone(timedelta(hours=9)))
    run_dir = args.output_dir.resolve() / started_at.strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("결과 저장 위치: %s", run_dir)
    logging.info("데이터와 학습 장치를 준비합니다. 중단은 Ctrl+C입니다.")

    train_dataset, val_dataset = load_mnist(seed=args.seed, augment=args.augment)
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size=args.batch_size, seed=args.seed,
    )
    device, model, criterion, optimizer = prepare_training(args.lr, args.seed, args.weight_decay)
    # cosine: 매 epoch 끝에 학습률을 줄여 마지막 epoch에서 0에 가깝게 만듦.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.scheduler == "cosine" else None
    )
    optimizer_name = type(optimizer).__name__
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    config = {
        "started_at": started_at.isoformat(),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "seed": args.seed,
        "optimizer": optimizer_name, "weight_decay": args.weight_decay,
        "scheduler": args.scheduler, "augment": args.augment,
        "log_interval": args.log_interval,
        "train_size": len(train_dataset), "val_size": len(val_dataset),
        "model": "MnistCNN: Conv 1->8->16, FC 784->32->10",
        "input": "NCHW, float32, pixel/255, " + (
            "train augmentation: rotate ±10deg, translate ±10%, scale 0.9-1.1"
            if args.augment else "no augmentation"
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device), "device_name": device_name,
        "torch_version": str(torch.__version__),
        "selection": "highest validation accuracy; lower validation loss breaks ties",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logging.info("장치: %s | 모델 파라미터: %s개", device_name, f"{config['parameters']:,}")
    logging.info(
        "학습 %s장 / 검증 %s장 | batch %d | %s lr %g, weight decay %g | scheduler %s | 증강 %s | %d epoch",
        f"{len(train_dataset):,}", f"{len(val_dataset):,}",
        args.batch_size, optimizer_name, args.lr, args.weight_decay,
        args.scheduler, "사용" if args.augment else "없음", args.epochs,
    )

    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_path = run_dir / "best.pt"
    training_start = perf_counter()
    fields = ["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds", "is_best"]
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        history_file.flush()

        for epoch in range(1, args.epochs + 1):
            epoch_start = perf_counter()
            lr = optimizer.param_groups[0]["lr"]  # 이번 epoch에 사용하는 학습률.
            logging.info("\nEpoch %d/%d | lr %.6g", epoch, args.epochs, lr)
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device, args.log_interval,
            )
            if scheduler is not None:
                scheduler.step()
            val_loss, val_accuracy = validate(
                model, val_loader, criterion, device, args.log_interval,
            )
            if not all(math.isfinite(v) for v in (train_loss, train_accuracy, val_loss, val_accuracy)):
                raise RuntimeError("손실 또는 정확도에 NaN/Inf가 발생했습니다. 로그를 확인하세요.")

            # 공식 test는 선택에 사용하지 않음. 검증 정확도가 같으면 손실로 비교.
            is_best = (val_accuracy, -val_loss) > (best_accuracy, -best_loss)
            if is_best:
                checkpoint = {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch, "val_accuracy": val_accuracy,
                    "val_loss": val_loss, "config": config,
                }
                # 저장 도중 중단되더라도 이전 best.pt를 보존하도록 임시 파일에서 교체.
                temporary_path = run_dir / "best.tmp"
                torch.save(checkpoint, temporary_path)
                temporary_path.replace(best_path)
                best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
                logging.info("  최고 모델 저장: best.pt (검증 정확도 %.2f%%)", 100 * best_accuracy)

            seconds = perf_counter() - epoch_start
            writer.writerow(dict(zip(fields, [
                epoch, lr, train_loss, train_accuracy, val_loss, val_accuracy, seconds, is_best,
            ])))
            history_file.flush()  # 실행 중에도 완료된 epoch 기록을 열어볼 수 있음.
            remaining = (perf_counter() - training_start) / epoch * (args.epochs - epoch)
            logging.info(
                "Epoch %d 완료 | train loss %.4f / acc %.2f%% | val loss %.4f / acc %.2f%%",
                epoch, train_loss, 100 * train_accuracy, val_loss, 100 * val_accuracy,
            )
            logging.info("소요 %.1f초 | 남은 학습·검증 예상 %.1f초 (최종 test 제외)", seconds, remaining)

    training_seconds = perf_counter() - training_start
    logging.info("\n학습 종료. 최고 모델(epoch %d)을 불러와 공식 test를 평가합니다.", best_epoch)
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataset = MNIST(
        root=PROJECT / "data",
        train=False, download=True, transform=ToTensor(),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loss, test_accuracy = validate(
        model, test_loader, criterion, device, args.log_interval, phase="최종 test",
    )
    if not math.isfinite(test_loss) or not math.isfinite(test_accuracy):
        raise RuntimeError("최종 test 결과에 NaN/Inf가 발생했습니다. 로그를 확인하세요.")
    results = {
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "best_val_accuracy": best_accuracy,
        "test_size": len(test_dataset), "test_loss": test_loss,
        "test_accuracy": test_accuracy, "training_seconds": training_seconds,
        "checkpoint": "best.pt",
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logging.info("최종 test | loss %.4f | accuracy %.2f%%", test_loss, 100 * test_accuracy)
    logging.info("완료. 가중치·설정·학습 기록·최종 결과: %s", run_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n사용자가 중단했습니다. 이미 저장된 최고 모델과 완료 epoch 기록은 유지됩니다.")
        sys.exit(130)
    except Exception:
        logging.exception("실행 중 오류가 발생했습니다.")
        sys.exit(1)
