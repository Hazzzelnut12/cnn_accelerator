# INT8 정수 규격 (v1)

RTL·ARM C가 비트 단위로 맞춰야 하는 계산 규칙. 기준 구현은 [quant.py](../../quant.py), 파라미터는 [models/mnist_int8_v1](../../models/mnist_int8_v1/).

## 1. 수치 형식

| 대상 | 형식 | 결정 — 근거 |
|---|---|---|
| weight | signed INT8, −127~127 | 대칭 범위 — 0 기준이 정확히 0, −128을 쓰지 않아 부호 반전에도 안전 |
| weight scale | 층별 1개 (`max\|w\| / 127`) | INT8 손실 0.00%p라 채널별 scale 불필요 |
| activation | signed INT8, ReLU 뒤라 0~127 | unsigned(0~255)와 정확도 차이 없음 → 모든 층을 signed 곱셈기 하나로 통일 ([비교](../analysis/int8-comparison.json)) |
| 입력 | 픽셀 0~255 → `(254·p + 255) // 510` (0~127) | 픽셀×127/255의 0.5 올림 반올림. ARM이 256칸 변환표([input_lut.mem](../../models/mnist_int8_v1/mem/input_lut.mem))로 처리 |
| bias | INT32, scale = 입력 scale × weight scale | 누산값에 그대로 더할 수 있음 |
| calibration | validation 최댓값 | 99.99 백분위와 차이 없음 → 잘라내는 값이 없는 단순한 쪽 |

## 2. 층 계산 순서

```text
acc = Σ w·x + b                       (정수 누산, bias 포함)
acc = max(acc, 0)                      (ReLU 먼저 → 이후 음수 없음)
y   = (acc · M + 2^(n−1)) >> n         (재양자화: 0.5 올림 반올림)
y   = min(y, 127)                      (포화)
```

- **ReLU를 재양자화보다 먼저** — 근거: 음수의 반올림·shift 방향 문제가 사라짐
- **MaxPool은 재양자화 뒤 8bit 값으로** — 근거: 재양자화는 증가 함수라 결과가 같고, 8bit 비교가 더 쌈
- **FC2는 재양자화 없이 INT32 logit 출력** — 근거: argmax에는 정수 그대로 충분, Softmax는 ARM에서 `logit × 0.001764`
- **Conv**: 3×3, stride 1, padding 1 (경계 밖은 0). 항 순서 `[c][ky][kx]`

## 3. 재양자화 파라미터

M은 16bit (2¹⁵ ≤ M < 2¹⁶), 실수 배율 m ≈ M / 2ⁿ.

| 층 | M | n | 실수 배율 m |
|---|---:|---:|---:|
| Conv1 | 34,106 | 24 | 0.00203 |
| Conv2 | 59,580 | 24 | 0.00355 |
| FC1 | 62,655 | 25 | 0.00187 |

## 4. bit 폭

| 층 | 누산값 관측 최대 | 최악 상한 | 결정 |
|---|---:|---:|---|
| Conv1 | 17bit | 18bit | 누산기 24bit — 근거: 모든 층의 최악 상한(22bit) 이상 |
| Conv2 | 18bit | 20bit | |
| FC1 | 18bit | 22bit | |
| FC2 | 16bit | 18bit | |

- **최악 상한** = 출력 채널별 Σ\|w\| × 127 + \|b\|의 최댓값. 어떤 입력에도 넘지 않음
- **재양자화 곱셈** acc(≤22bit, 음수 없음) × M(16bit) = 최대 38bit → DSP48 하나(25×18 곱셈기)에 들어감

## 5. 결과

| | float | INT8 |
|---|---:|---:|
| validation | 98.98% | 98.96% |
| test | 99.04% | 99.04% (손실 0.00%p, 예측 일치 99.89%) |

INT8 0 비율 (validation): Conv1 입력 80.9%, Conv2 입력 47.3%, FC1 입력 59.3%, FC2 입력 57.1%

## 6. 파일 형식

- **weight·bias** ([mem/](../../models/mnist_int8_v1/mem/)): 한 줄에 값 하나, 2의 보수 hex. INT8은 2자리, INT32는 8자리. `$readmemh`로 읽음
- **순서**: Conv weight `[out][in][ky][kx]`, FC weight `[out][in]`, FC1 입력은 Pool2 출력의 CHW 순서
- **golden trace** (`build/golden/`, Git 제외): 입력마다 `input`, `conv1_acc`, `conv1_out`, `pool1`, `conv2_acc`, `conv2_out`, `pool2`, `fc1_acc`, `fc1_out`, `logits`. test 20장 + 경계 입력 3개(전부 0, 전부 127, 무작위)
- **BRAM 묶음 배치**: RTL 구조 설계(P4a)에서 위 파일을 변환해 결정
- **재생성**: `scripts/quantize.py --save signed:max` → `scripts/export_golden.py`
