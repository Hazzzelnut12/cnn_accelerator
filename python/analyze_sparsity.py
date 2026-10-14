"""확정 float 모델의 activation 희소성과 dense/skip 예상 cycle을 validation으로 분석.

- 데이터: train.py와 같은 seed의 validation 5,000장. test는 사용하지 않음(정책 결정 전).
- 0 판정: float의 정확한 0, 그리고 층별 max 기준 INT8 양자화를 가정한 추정값.
- cycle 모델: 출력 채널 병렬 MAC P개, 한 cycle에 입력 항 1개를 P개 출력 채널에 공급.
  padding 위치는 모든 방식에서 제외(고정 padding 제거를 희소성 이득으로 세지 않음).
  RTL 실측이 아닌 예상치이며 window 읽기·stall·가중치 공급 충돌은 포함하지 않음.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import MnistCNN, load_mnist

PROJECT = Path(__file__).resolve().parent.parent


# 블록 1: 각 관측 위치의 activation 계산 (train.py MnistCNN.forward와 같은 순서)
def activations(model: MnistCNN, x: torch.Tensor) -> dict[str, torch.Tensor]:
    relu1 = model.relu1(model.conv1(x))
    pool1 = model.pool1(relu1)
    relu2 = model.relu2(model.conv2(pool1))
    pool2 = model.pool2(relu2)
    fc1 = model.relu3(model.fc1(model.flatten(pool2)))
    return {"input": x, "relu1": relu1, "pool1": pool1, "relu2": relu2, "pool2": pool2, "fc1": fc1}


# 블록 2: 3×3 window별 유효 입력 항 수(padding 제외)와 0이 아닌 항 수
def window_counts(nonzero: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = nonzero.size(1)
    kernel = torch.ones(1, channels, 3, 3, device=nonzero.device)
    terms = F.conv2d(torch.ones_like(nonzero[:1]), kernel, padding=1)  # 경계 window는 항이 적음
    nnz = F.conv2d(nonzero, kernel, padding=1)
    return terms.flatten(1), nnz.flatten(1)  # (1, H*W), (N, H*W)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    v = values.double().flatten()
    q = torch.quantile(v, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return {"mean": v.mean().item(), "p5": q[0].item(), "p50": q[1].item(), "p95": q[2].item(),
            "min": v.min().item(), "max": v.max().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "models" / "mnist_float_v1" / "best.pt")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "sparsity-mnist_float_v1.json")
    parser.add_argument("--mac", type=int, default=8, help="출력 채널 병렬 MAC 수 P (기본: 8)")
    parser.add_argument("--overhead", type=int, nargs="+", default=[0, 2], help="skip window당 추가 cycle 가정")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MnistCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, val_dataset = load_mnist(seed=checkpoint["config"]["seed"])
    images = torch.cat([x for x, _ in DataLoader(val_dataset, batch_size=1000)])

    with torch.inference_mode():
        acts = activations(model, images)

    # 블록 3: 층별 0 비율 (float, INT8 추정)
    layers = {}
    for name, a in acts.items():
        per_image = (a == 0).flatten(1).float().mean(1)
        scale = a.max().item() / 127  # 층별 최대값 calibration 가정. 실제 PTQ 규격은 P2에서 확정.
        int8_zero = (torch.round(a / scale) == 0).flatten(1).float().mean(1)
        entry = {
            "shape": list(a.shape[1:]),
            "float_zero_ratio": quantiles(per_image),
            "int8_zero_ratio_estimate": quantiles(int8_zero),
            "max_value": a.max().item(),
        }
        if a.dim() == 4 and name != "input":
            entry["per_channel_float_zero_ratio"] = (a == 0).float().mean((0, 2, 3)).tolist()
        layers[name] = entry

    # 블록 4: Conv 입력 window의 유효 항 분포와 cycle 추정
    P = args.mac
    conv_inputs = {"conv1": (acts["input"], 8), "conv2": (acts["pool1"], 16)}
    windows, cycles = {}, {"dense": {}, "window_skip": {}}
    for ov in args.overhead:
        cycles[f"skip_c{ov}"], cycles[f"adaptive_c{ov}"] = {}, {}
    for kind, quantized in (("float", False), ("int8_est", True)):
        for name, (inp, out_ch) in conv_inputs.items():
            if quantized:
                scale = inp.max().item() / 127
                nonzero = (torch.round(inp / scale) != 0).float()
            else:
                nonzero = (inp != 0).float()
            terms, nnz = window_counts(nonzero)
            passes = math.ceil(out_ch / P)
            if kind == "float":
                windows[name] = {
                    "terms_per_window_max": int(terms.max().item()),
                    "nonzero_fraction_of_terms": quantiles((nnz / terms).flatten()),
                    "all_zero_window_ratio": quantiles((nnz == 0).float().mean(1)),
                }
            key = f"{name}_{kind}"
            cycles["dense"][key] = quantiles(terms.sum().expand(nnz.size(0)) * passes)
            cycles["window_skip"][key] = quantiles(torch.where(nnz == 0, 1.0, terms).sum(1) * passes)
            for ov in args.overhead:
                cycles[f"skip_c{ov}"][key] = quantiles((nnz + ov).sum(1) * passes)
                cycles[f"adaptive_c{ov}"][key] = quantiles(torch.minimum(terms, nnz + ov).sum(1) * passes)
        # FC: 0이 아닌 입력만 읽어 해당 가중치 열을 출력 묶음에 누산(input-stationary)
        for name, inp, out_n in (("fc1", acts["pool2"], 32), ("fc2", acts["fc1"], 10)):
            flat = inp.flatten(1)
            if quantized:
                nonzero = torch.round(flat / (flat.max().item() / 127)) != 0
            else:
                nonzero = flat != 0
            passes = math.ceil(out_n / P)
            key = f"{name}_{kind}"
            dense = torch.full((flat.size(0),), float(flat.size(1) * passes))
            sparse = nonzero.float().sum(1) * passes
            cycles["dense"][key] = quantiles(dense)
            cycles["window_skip"][key] = quantiles(dense)
            for ov in args.overhead:
                cycles[f"skip_c{ov}"][key] = quantiles(sparse)
                cycles[f"adaptive_c{ov}"][key] = quantiles(torch.minimum(dense, sparse))

    report = {
        "checkpoint": str(args.checkpoint.relative_to(PROJECT)) if args.checkpoint.is_relative_to(PROJECT) else str(args.checkpoint),
        "data": f"validation {images.size(0)} images (seed {checkpoint['config']['seed']})",
        "cycle_model": f"P={P} output-channel MACs, 1 input term/cycle, padding excluded, estimate only",
        "layers": layers,
        "conv_windows": windows,
        "cycles": cycles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 블록 5: 요약 출력
    print(f"데이터: {report['data']} | 모델: {report['checkpoint']}")
    print("층별 0 비율 평균 (float / INT8 추정)")
    for name, e in layers.items():
        print(f"  {name:6s} {100 * e['float_zero_ratio']['mean']:5.1f}% / {100 * e['int8_zero_ratio_estimate']['mean']:5.1f}%"
              f"  (float 이미지별 p5~p95 {100 * e['float_zero_ratio']['p5']:.1f}~{100 * e['float_zero_ratio']['p95']:.1f}%)")
    for name, w in windows.items():
        print(f"  {name} window: 유효 항 비율 평균 {100 * w['nonzero_fraction_of_terms']['mean']:.1f}%, "
              f"전부 0인 window {100 * w['all_zero_window_ratio']['mean']:.1f}%")
    print(f"이미지당 예상 cycle 평균 (P={P})")
    for kind in ("float", "int8_est"):
        print(f"  [{kind}]")
        for mode, table in cycles.items():
            per_layer = {k.split("_")[0]: v["mean"] for k, v in table.items() if k.endswith(kind)}
            total = sum(per_layer.values())
            parts = " ".join(f"{k} {v:,.0f}" for k, v in per_layer.items())
            print(f"    {mode:12s} 합계 {total:8,.0f} | {parts}")
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
