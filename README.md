# CNN accelerator

Zynq-7020 보드에서 손글씨 숫자열을 실시간으로 판독. LeNet 계열 CNN을 직접 작성한 Verilog로 가속.

- **계획:** [계획.md](계획.md) · **진행 기록:** [개발기록.md](개발기록.md)
- **확정 모델:** float [mnist_float_v1](models/mnist_float_v1/) (test 99.04%) → INT8 [mnist_int8_v1](models/mnist_int8_v1/) (test 99.04%), 규격 [int8-spec](docs/spec/int8-spec.md)
- **진행:** P0~P3 ✅ · 다음 P4a RTL 구조 설계

## 시작하기

필요 조건: Windows x64, NVIDIA GPU, 인터넷 (최초 설치 시)

**1. 환경 구성** — `.venv` 생성, 성공하면 마지막 출력에 `"status": "PASS"`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

**2. 확정 모델 확인** — 학습 없이 test만 평가, 성공하면 `→ 일치`

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate.py
```

이후 작업(INT8·RTL)은 직접 학습한 모델이 아니라 `models/mnist_float_v1/best.pt` 기준.

## 학습

```powershell
# 최종 모델과 같은 설정
.\.venv\Scripts\python.exe -u .\train.py --epochs 30 --augment --scheduler cosine --weight-decay 1e-3
```

| 옵션 | 기본값 | 내용 |
|---|---|---|
| `--epochs` / `--batch-size` / `--lr` | 20 / 64 / 0.001 | |
| `--augment` | 꺼짐 | 학습 데이터만 회전·이동·크기 증강 |
| `--scheduler cosine` | none | 학습률을 마지막 epoch까지 점차 감소 |
| `--weight-decay` | 0 (Adam) | 0보다 크면 AdamW |

- **결과 위치:** `artifacts/mnist/<실행 시각>/` — `best.pt`, `config.json`, `history.csv`, `train.log`, `results.json`
- **공유:** `artifacts/`는 Git 제외. 확정한 모델만 `models/<이름>/`으로 복사해서 commit
- **희소성 분석:** `.\.venv\Scripts\python.exe .\scripts\analyze_sparsity.py` → `docs/analysis/`

## INT8 정수 모델

```powershell
.\.venv\Scripts\python.exe .\scripts\quantize.py --save signed:max   # 설정 4개 비교 + 선택 설정 저장
.\.venv\Scripts\python.exe .\scripts\export_golden.py               # hex export + golden trace + 재계산 검증
```

- 성공 기준: 마지막 줄 `PASS`
- golden trace는 `build/golden/`에 생성 (Git 제외, 위 명령으로 재생성)

## 폴더

| 경로 | 내용 |
|---|---|
| `train.py` | 모델·데이터·학습·평가 |
| `quant.py` | INT8 정수 reference (RTL이 맞출 정답) |
| `models/` | 확정 모델 (Git 포함) |
| `scripts/` | 환경 설치·검증, 평가, 희소성 분석 |
| `docs/` | 검증 결과, 분석 결과, 조사 자료 |
| `artifacts/`, `data/`, `.venv/` | 실험 결과·데이터셋·환경 (Git 제외) |
