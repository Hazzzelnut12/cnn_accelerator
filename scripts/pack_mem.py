"""정수 모델을 가속기 메모리 배치(P4a 구조)에 맞게 묶어서 RTL용 파일 생성.

- weights.mem: 한 줄 = 64bit word = 출력 채널 8개의 weight. word[g][t] = {w[8g+7][t], ..., w[8g][t]}, 하위 byte가 PE0
  층 순서 Conv1 → Conv2 → FC1 → FC2, 층 안에서는 묶음 g → 항 t 순서. Conv 항 t = (c, ky, kx), FC 항 t = 입력 index
- bias.mem: 한 줄 = INT32 bias 1개. 층 순서, 층 안에서는 출력 채널 순서
- cnn_params.vh: 재양자화 M·n, 메모리 시작 주소 등 RTL 상수
만든 뒤 다시 풀어서 원래 weight·bias와 같은지 확인.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import quant  # noqa: E402

P = 8  # PE 수 = 한 word에 묶는 출력 채널 수
BIAS_BITS = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "mnist_int8_v1")
    parser.add_argument("--rtl-dir", type=Path, default=PROJECT / "rtl")
    args = parser.parse_args()

    params = quant.load_params(args.model_dir)
    words, biases, wbase, bbase, info = [], [], {}, {}, {}
    for name, _, _ in quant.LAYERS:
        L = params["layers"][name]
        w = L["w"].reshape(L["w"].shape[0], -1)  # (출력, 항 K)
        out_n, K = w.shape
        groups = -(-out_n // P)
        wpad = np.zeros((groups * P, K), np.int64)
        wpad[:out_n] = w  # FC2처럼 8의 배수가 아니면 남는 PE는 weight 0
        wbase[name], bbase[name] = len(words), len(biases)
        for g in range(groups):
            for t in range(K):
                lanes = wpad[g * P:(g + 1) * P, t]
                words.append(sum((int(v) & 0xFF) << (8 * i) for i, v in enumerate(lanes)))
        biases.extend(int(b) for b in L["b"])
        info[name] = {"out": out_n, "K": K, "groups": groups}

    mem_dir, inc_dir = args.rtl_dir / "mem", args.rtl_dir / "include"
    mem_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "weights.mem").write_text("".join(f"{w:016x}\n" for w in words), encoding="ascii")
    (mem_dir / "bias.mem").write_text("".join(f"{b & 0xFFFFFFFF:08x}\n" for b in biases), encoding="ascii")

    bias_max = max(abs(b) for b in biases)
    lines = [
        "// 자동 생성: scripts/pack_mem.py. 직접 수정하지 말 것.",
        f"// 원본: {args.model_dir.relative_to(PROJECT).as_posix()}/spec.json",
        f"`define CNN_P            {P}",
        "`define CNN_DATA_W       8",
        "`define CNN_ACC_W        24",
        f"`define CNN_M_W          {quant.M_BITS}",
        f"`define CNN_QMAX         {params['qmax']}",
        f"`define CNN_WROM_DEPTH   {len(words)}",
        f"`define CNN_BIAS_DEPTH   {len(biases)}",
    ]
    for name, _, dst in quant.LAYERS:
        u, L = name.upper(), params["layers"][name]
        lines += [f"`define {u}_WBASE     {wbase[name]}", f"`define {u}_BBASE     {bbase[name]}",
                  f"`define {u}_K         {info[name]['K']}", f"`define {u}_GROUPS    {info[name]['groups']}"]
        if dst is not None:
            lines += [f"`define {u}_M         {L['M']}", f"`define {u}_SHIFT     {L['n']}"]
    (inc_dir / "cnn_params.vh").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 다시 풀어서 확인
    read = [int(x, 16) for x in (mem_dir / "weights.mem").read_text(encoding="ascii").split()]
    ok = True
    for name, _, _ in quant.LAYERS:
        L = params["layers"][name]
        w = L["w"].reshape(L["w"].shape[0], -1)
        for o in range(w.shape[0]):
            g, lane = divmod(o, P)
            for t in range(w.shape[1]):
                byte = (read[wbase[name] + g * w.shape[1] + t] >> (8 * lane)) & 0xFF
                ok &= (byte - 256 if byte >= 128 else byte) == w[o, t]
    rb = [int(x, 16) for x in (mem_dir / "bias.mem").read_text(encoding="ascii").split()]
    ok &= [b - (1 << 32) if b >= 1 << 31 else b for b in rb] == biases

    print(f"weights.mem: {len(words)} word × 64 bit "
          + " ".join(f"{n} {wbase[n]}+{info[n]['groups'] * info[n]['K']}" for n, _, _ in quant.LAYERS))
    print(f"bias.mem: {len(biases)}개, 최대 |bias| {bias_max} ({bias_max.bit_length() + 1} bit)")
    print(f"cnn_params.vh: {inc_dir / 'cnn_params.vh'}")
    if not ok:
        raise SystemExit("FAIL: 풀어낸 값이 원본과 다름")
    print("PASS")


if __name__ == "__main__":
    main()
