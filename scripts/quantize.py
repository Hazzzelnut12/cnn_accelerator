"""float 모델을 INT8로 양자화하고 activation 형식·calibration 방식별로 비교. 선택한 설정을 저장.

- calibration·설정 선택: validation 5,000장. test 10,000장은 결과 보고용.
- 비교: activation signed(0~127) / unsigned(0~255) × calibration max / p99.99
- 측정: 정확도, float와 예측 일치율, 층별 누산기 최댓값과 필요한 bit 수, 실제 INT8 0 비율
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import quant  # noqa: E402
from train import MnistCNN, load_mnist  # noqa: E402

ACC_LAYERS = ["conv1_acc", "conv2_acc", "fc1_acc", "logits"]
ZERO_POINTS = {"input": "Conv1 입력", "pool1": "Conv2 입력", "pool2": "FC1 입력", "fc1_out": "FC2 입력"}


def signed_bits(v: int) -> int:
    """절댓값 v를 담는 signed 정수의 bit 수."""
    return int(v).bit_length() + 1


def worst_case(params: dict) -> dict[str, int]:
    """입력이 모두 최댓값일 때 가능한 누산값 절댓값의 상한 (출력 채널별 Σ|w|·qmax + |b| 중 최대)."""
    out, qmax = {}, params["qmax"]
    for name, key in zip(["conv1", "conv2", "fc1", "fc2"], ACC_LAYERS):
        L = params["layers"][name]
        w = L["w"].reshape(L["w"].shape[0], -1)
        out[key] = int((np.abs(w).sum(1) * qmax + np.abs(L["b"])).max())
    return out


def evaluate(params: dict, pixels: np.ndarray, labels: np.ndarray, batch: int = 1000) -> dict:
    q = quant.quantize_input(pixels.reshape(len(pixels), -1), params["variant"])
    preds, acc_max = [], {k: 0 for k in ACC_LAYERS}
    zeros = {k: 0 for k in ZERO_POINTS}
    sizes = {}
    for i in range(0, len(q), batch):
        logits, t = quant.forward(params, q[i:i + batch], trace=True)
        preds.append(logits.argmax(1))
        for k in ACC_LAYERS:
            acc_max[k] = max(acc_max[k], int(np.abs(t[k]).max()))
        for k in ZERO_POINTS:
            zeros[k] += int((t[k] == 0).sum())
            sizes[k] = sizes.get(k, 0) + t[k].size
    preds = np.concatenate(preds)
    return {"accuracy": float((preds == labels).mean()), "preds": preds, "acc_max": acc_max,
            "zero_ratio": {k: zeros[k] / sizes[k] for k in ZERO_POINTS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "mnist_float_v1")
    parser.add_argument("--report", type=Path, default=PROJECT / "docs" / "analysis" / "int8-comparison.json")
    parser.add_argument("--save", metavar="VARIANT:METHOD", help="이 설정을 저장 (예: unsigned:max)")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "models" / "mnist_int8_v1")
    args = parser.parse_args()

    checkpoint = torch.load(args.model_dir / "best.pt", map_location="cpu", weights_only=True)
    model = MnistCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sd = checkpoint["model_state_dict"]

    _, val_dataset = load_mnist(seed=checkpoint["config"]["seed"])
    val_images, val_labels = map(torch.cat, zip(*[(x, y) for x, y in DataLoader(val_dataset, batch_size=1000)]))
    val_pixels = torch.round(val_images * 255).to(torch.uint8).numpy()  # ToTensor 이전의 원래 픽셀로 복원
    test = MNIST(root=PROJECT / "data", train=False, download=True)
    test_pixels, test_labels = test.data.numpy(), test.targets.numpy()
    with torch.inference_mode():
        float_test = model(torch.from_numpy(test_pixels).float().div(255).unsqueeze(1)).argmax(1).numpy()
    float_acc = float((float_test == test_labels).mean())

    calibration = {m: quant.calibrate(model, val_images, m) for m in ("max", "p99.99")}
    rows = {}
    for variant in ("signed", "unsigned"):
        for method in ("max", "p99.99"):
            params = quant.quantize_model(sd, calibration[method], variant)
            val = evaluate(params, val_pixels, val_labels.numpy())
            tst = evaluate(params, test_pixels, test_labels)
            bound = worst_case(params)
            observed = {k: max(val["acc_max"][k], tst["acc_max"][k]) for k in ACC_LAYERS}
            rows[f"{variant}:{method}"] = {
                "val_accuracy": val["accuracy"], "test_accuracy": tst["accuracy"],
                "test_loss_pp": 100 * (float_acc - tst["accuracy"]),
                "test_agreement_with_float": float((tst["preds"] == float_test).mean()),
                "acc_abs_max_observed": observed,
                "acc_bits_observed": {k: signed_bits(v) for k, v in observed.items()},
                "acc_abs_bound": bound,
                "acc_bits_bound": {k: signed_bits(v) for k, v in bound.items()},
                "requant": {n: {"M": params["layers"][n]["M"], "n": params["layers"][n]["n"]} for n in ("conv1", "conv2", "fc1")},
                "zero_ratio_val": val["zero_ratio"],
            }

    report = {"float_test_accuracy": float_acc, "calibration_act_max": calibration, "configs": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"float test {100 * float_acc:.2f}%")
    print(f"{'설정':16s} {'val':>7s} {'test':>7s} {'손실':>6s} {'일치':>7s}  누산기 bit (관측/상한) conv1 conv2 fc1 fc2")
    for key, r in rows.items():
        bits = " ".join(f"{r['acc_bits_observed'][k]}/{r['acc_bits_bound'][k]}" for k in ACC_LAYERS)
        print(f"{key:16s} {100 * r['val_accuracy']:6.2f}% {100 * r['test_accuracy']:6.2f}% "
              f"{r['test_loss_pp']:5.2f}p {100 * r['test_agreement_with_float']:6.2f}%  {bits}")
    print("INT8 0 비율 (validation): " + " | ".join(
        f"{key}: " + ", ".join(f"{ZERO_POINTS[k]} {100 * v:.1f}%" for k, v in r["zero_ratio_val"].items())
        for key, r in rows.items() if key.endswith(":max")))
    print(f"비교 결과: {args.report}")

    if args.save:
        variant, method = args.save.split(":")
        params = quant.quantize_model(sd, calibration[method], variant)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {f"{n}_{k}": params["layers"][n][k] for n, _, _ in quant.LAYERS for k in ("w", "b")}
        np.savez(args.out_dir / "params.npz", **arrays)
        spec = {
            "source_model": str((args.model_dir / "best.pt").relative_to(PROJECT)).replace("\\", "/"),
            "source_sha256": hashlib.sha256((args.model_dir / "best.pt").read_bytes()).hexdigest(),
            "variant": variant, "qmax": params["qmax"], "weight_max": quant.WMAX, "multiplier_bits": quant.M_BITS,
            "calibration": method, "act_max": params["act_max"],
            "layers": {n: {k: v for k, v in params["layers"][n].items() if k not in ("w", "b")} | {
                "w_shape": list(params["layers"][n]["w"].shape)} for n, _, _ in quant.LAYERS},
            "results": rows[args.save],
        }
        (args.out_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장: {args.out_dir} ({args.save})")


if __name__ == "__main__":
    main()
