"""정수 모델을 RTL용 hex(.mem) 파일로 내보내고, 층 경계 golden trace를 만든 뒤 다시 읽어 재계산 검증.

- 값은 논리 순서 그대로 한 줄에 하나: weight [out][in][ky][kx], FC weight [out][in], feature map CHW.
  BRAM 묶음 배치는 RTL 구조 설계(P4a)에서 이 파일을 변환해 만듦.
- 표기: 2의 보수 hex. INT8은 2자리, INT32는 8자리. Verilog $readmemh로 바로 읽을 수 있음.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import quant  # noqa: E402

# 텐서별 hex 자릿수: 누산값·bias·logit은 INT32, 나머지는 8bit
WIDE = {"conv1_acc", "conv2_acc", "fc1_acc", "logits"}


def write_mem(path: Path, values: np.ndarray, digits: int) -> None:
    mask = (1 << (4 * digits)) - 1
    path.write_text("".join(f"{int(v) & mask:0{digits}x}\n" for v in values.ravel()), encoding="ascii")


def read_mem(path: Path, digits: int) -> np.ndarray:
    raw = np.array([int(line, 16) for line in path.read_text(encoding="ascii").split()], dtype=np.int64)
    sign = 1 << (4 * digits - 1)
    return np.where(raw >= sign, raw - (1 << (4 * digits)), raw)  # 2의 보수 → 부호 있는 정수


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "mnist_int8_v1")
    parser.add_argument("--golden-dir", type=Path, default=PROJECT / "build" / "golden")
    parser.add_argument("--count", type=int, default=20, help="golden trace를 만들 test 이미지 수 (앞에서부터)")
    args = parser.parse_args()

    params = quant.load_params(args.model_dir)
    spec = json.loads((args.model_dir / "spec.json").read_text(encoding="utf-8"))

    # 블록 1: weight·bias·입력 변환표 export
    mem_dir = args.model_dir / "mem"
    mem_dir.mkdir(exist_ok=True)
    for name, _, _ in quant.LAYERS:
        write_mem(mem_dir / f"{name}_w.mem", params["layers"][name]["w"], 2)
        write_mem(mem_dir / f"{name}_b.mem", params["layers"][name]["b"], 8)
    # 픽셀(0~255) → 모델 입력 정수 변환표. ARM이 전처리 마지막에 사용.
    write_mem(mem_dir / "input_lut.mem", quant.quantize_input(np.arange(256), params["variant"]), 2)

    # 블록 2: golden trace — test 이미지 + 경계 조건 입력
    test = MNIST(root=PROJECT / "data", train=False, download=True)
    qmax = params["qmax"]
    cases = [(f"{i:03d}_test{i}", quant.quantize_input(test.data[i].numpy().ravel(), params["variant"]), int(test.targets[i]))
             for i in range(args.count)]
    rng = np.random.default_rng(0)
    cases += [("zero", np.zeros(784, np.int64), None),          # 전부 0: bias만 남는 경우
              ("max", np.full(784, qmax, np.int64), None),       # 전부 최댓값: 누산기·포화 경계
              ("random", rng.integers(0, qmax + 1, 784), None)]  # 무작위: 희소하지 않은 입력
    if args.golden_dir.exists():
        shutil.rmtree(args.golden_dir)
    index = []
    for case, q_in, label in cases:
        logits, trace = quant.forward(params, q_in[None, :], trace=True)
        case_dir = args.golden_dir / case
        case_dir.mkdir(parents=True)
        for key, tensor in trace.items():
            write_mem(case_dir / f"{key}.mem", tensor[0], 8 if key in WIDE else 2)
        index.append({"case": case, "label": label, "pred": int(logits[0].argmax()),
                      "shapes": {k: list(v.shape[1:]) for k, v in trace.items()}})
    (args.golden_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # 블록 3: 재계산 검증 — export한 파일만으로 다시 계산해서 golden과 비교
    reloaded = {"variant": params["variant"], "qmax": qmax, "layers": {}}
    for name, _, _ in quant.LAYERS:
        L = params["layers"][name]
        reloaded["layers"][name] = dict(L, w=read_mem(mem_dir / f"{name}_w.mem", 2).reshape(L["w"].shape),
                                        b=read_mem(mem_dir / f"{name}_b.mem", 8))
    mismatches = 0
    for entry in index:
        case_dir = args.golden_dir / entry["case"]
        _, trace = quant.forward(reloaded, read_mem(case_dir / "input.mem", 2)[None, :], trace=True)
        for key, tensor in trace.items():
            golden = read_mem(case_dir / f"{key}.mem", 8 if key in WIDE else 2)
            mismatches += int((golden != tensor[0].ravel()).sum())
    q_test = quant.quantize_input(test.data.numpy().reshape(len(test.data), -1), params["variant"])
    accuracy = float((quant.forward_batched(reloaded, q_test).argmax(1) == test.targets.numpy()).mean())
    expected = spec["results"]["test_accuracy"]

    print(f"weight·bias: {mem_dir} ({sum(1 for _ in mem_dir.glob('*.mem'))}개 파일)")
    print(f"golden trace: {args.golden_dir} ({len(index)}개 입력, 입력마다 텐서 {len(trace)}개)")
    print(f"재계산 검증: golden 불일치 {mismatches}개 | 재로드 모델 test {100 * accuracy:.2f}% (기록 {100 * expected:.2f}%)")
    if mismatches or accuracy != expected:
        raise SystemExit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
