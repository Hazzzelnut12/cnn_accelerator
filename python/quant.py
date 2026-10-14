"""INT8 정수 reference: 학습된 float 모델을 정수 연산만으로 계산하는 기준 구현.

RTL·ARM C가 비트 단위로 맞춰야 하는 정답. 추론 경로(forward)에는 float를 쓰지 않음.
float는 오프라인 파라미터 생성(calibrate, quantize_model)에서만 사용.
"""

import math

import numpy as np
import torch

# 블록 1: 수치 형식
# activation은 ReLU 뒤라 음수가 없음. signed는 0~127, unsigned는 0~255 사용.
QMAX = {"signed": 127, "unsigned": 255}
WMAX = 127   # weight: signed -127~127 대칭 (-128은 쓰지 않음)
M_BITS = 16  # 재양자화 multiplier 비트 수 (DSP 18bit 입력에 들어가는 크기)

# (층 이름, 입력 activation, 출력 activation). 출력이 None이면 재양자화 없이 INT32 logit.
LAYERS = [("conv1", "input", "relu1"), ("conv2", "relu1", "relu2"),
          ("fc1", "relu2", "relu3"), ("fc2", "relu3", None)]


# 블록 2: calibration — validation에서 층별 activation 크기 측정 (float, 오프라인)
@torch.inference_mode()
def calibrate(model, images: torch.Tensor, method: str = "max", batch: int = 1000) -> dict[str, float]:
    """method: "max"(최댓값) 또는 "p99.99"(0이 아닌 값의 99.99 백분위, 드문 큰 값은 포화시킴)."""
    model.eval()
    values = {"relu1": [], "relu2": [], "relu3": []}
    for start in range(0, images.size(0), batch):
        x = images[start:start + batch]
        r1 = model.relu1(model.conv1(x))
        r2 = model.relu2(model.conv2(model.pool1(r1)))
        r3 = model.relu3(model.fc1(model.flatten(model.pool2(r2))))
        for name, a in (("relu1", r1), ("relu2", r2), ("relu3", r3)):
            values[name].append(a[a > 0].numpy() if method != "max" else a.max().reshape(1).numpy())
    result = {"input": 1.0}  # 입력은 픽셀 ÷ 255라 최댓값이 정확히 1
    for name, chunks in values.items():
        v = np.concatenate(chunks)
        result[name] = float(v.max() if method == "max" else np.percentile(v, 99.99))
    return result


def requant_params(m: float) -> tuple[int, int]:
    """실수 배율 m을 M / 2^n으로 근사. M은 [2^15, 2^16) 범위의 16bit 정수."""
    n = M_BITS - 1 - math.floor(math.log2(m))
    M = round(m * 2**n)
    if M == 2**M_BITS:
        M, n = M // 2, n - 1
    return M, n


# 블록 3: 파라미터 양자화 (float → 정수, 오프라인)
def quantize_model(state_dict: dict, act_max: dict[str, float], variant: str) -> dict:
    qmax = QMAX[variant]
    s_act = {k: v / qmax for k, v in act_max.items()}  # 실수값 = scale × 정수값
    layers = {}
    for name, src, dst in LAYERS:
        w = state_dict[f"{name}.weight"].double().numpy()
        b = state_dict[f"{name}.bias"].double().numpy()
        s_w = float(np.abs(w).max() / WMAX)
        s_in = s_act[src]
        entry = {
            "w": np.clip(np.round(w / s_w), -WMAX, WMAX).astype(np.int64),
            "b": np.round(b / (s_in * s_w)).astype(np.int64),  # bias scale = 입력 scale × weight scale
            "s_w": s_w, "s_in": s_in,
        }
        if dst is None:
            entry["s_out"] = s_in * s_w  # logit 실수값 = s_out × 정수 logit (ARM Softmax용)
        else:
            entry["M"], entry["n"] = requant_params(s_in * s_w / s_act[dst])
            entry["s_out"] = s_act[dst]
        layers[name] = entry
    return {"variant": variant, "qmax": qmax, "act_max": act_max, "layers": layers}


def quantize_input(pixels: np.ndarray, variant: str) -> np.ndarray:
    """카메라·MNIST 픽셀(uint8 0~255) → 모델 입력 정수. signed는 픽셀×127/255를 0.5 올림 반올림."""
    p = pixels.astype(np.int64)
    return p if variant == "unsigned" else (2 * 127 * p + 255) // 510


# 블록 4: 정수 연산 (RTL이 그대로 구현할 식)
def requantize(acc: np.ndarray, M: int, n: int, qmax: int) -> np.ndarray:
    """누산값(bias 포함) → ReLU → ×M, 0.5 올림 반올림 후 n bit 오른쪽 shift → qmax 포화."""
    acc = np.maximum(acc, 0)
    return np.minimum((acc * M + (1 << (n - 1))) >> n, qmax)


def conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """stride 1, padding 1. x (N,C,H,W), w (O,C,3,3) → 누산값 (N,O,H,W), bias 포함."""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.stack([xp[:, :, ky:ky + H, kx:kx + W] for ky in range(3) for kx in range(3)], axis=2)
    acc = np.matmul(w.reshape(w.shape[0], C * 9), cols.reshape(N, C * 9, H * W))  # 항 순서 [c][ky][kx]
    return acc.reshape(N, -1, H, W) + b[None, :, None, None]


def maxpool2(x: np.ndarray) -> np.ndarray:
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def forward(params: dict, q_in: np.ndarray, trace: bool = False):
    """q_in (N,784) 정수 입력 → INT32 logit (N,10). trace=True면 층 경계 텐서도 반환."""
    L, qmax = params["layers"], params["qmax"]
    x = q_in.reshape(-1, 1, 28, 28).astype(np.int64)
    a1 = conv3x3(x, L["conv1"]["w"], L["conv1"]["b"])
    r1 = requantize(a1, L["conv1"]["M"], L["conv1"]["n"], qmax)
    p1 = maxpool2(r1)
    a2 = conv3x3(p1, L["conv2"]["w"], L["conv2"]["b"])
    r2 = requantize(a2, L["conv2"]["M"], L["conv2"]["n"], qmax)
    p2 = maxpool2(r2)
    a3 = p2.reshape(p2.shape[0], -1) @ L["fc1"]["w"].T + L["fc1"]["b"]  # flatten 순서 CHW
    r3 = requantize(a3, L["fc1"]["M"], L["fc1"]["n"], qmax)
    a4 = r3 @ L["fc2"]["w"].T + L["fc2"]["b"]
    if not trace:
        return a4
    return a4, {"input": x, "conv1_acc": a1, "conv1_out": r1, "pool1": p1,
                "conv2_acc": a2, "conv2_out": r2, "pool2": p2,
                "fc1_acc": a3, "fc1_out": r3, "logits": a4}


def forward_batched(params: dict, q_in: np.ndarray, batch: int = 1000) -> np.ndarray:
    return np.concatenate([forward(params, q_in[i:i + batch]) for i in range(0, len(q_in), batch)])


# 블록 5: 저장된 정수 모델 불러오기 (python/quantize.py --save 결과)
def load_params(model_dir) -> dict:
    import json
    from pathlib import Path
    model_dir = Path(model_dir)
    spec = json.loads((model_dir / "spec.json").read_text(encoding="utf-8"))
    arrays = np.load(model_dir / "params.npz")
    layers = {n: dict(spec["layers"][n], w=arrays[f"{n}_w"].astype(np.int64), b=arrays[f"{n}_b"].astype(np.int64))
              for n, _, _ in LAYERS}
    return {"variant": spec["variant"], "qmax": spec["qmax"], "act_max": spec["act_max"], "layers": layers}
