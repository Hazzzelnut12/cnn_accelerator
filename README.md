# CNN accelerator

- **목표:** MNIST 숫자를 분류하는 CNN을 학습하고, 직접 설계한 Verilog 가속기에서 실행
- **모델:** LeNet 파생 소형 CNN → INT8 양자화
- **대상 보드:** Zynq-7020
- **설계·작업 순서:** [계획.md](계획.md)
- **구현·실험 기록:** [개발기록.md](개발기록.md)
- **확정 모델:** [models/mnist_float_v1](models/mnist_float_v1/) · test 99.04%

## 처음 받은 경우 (팀원용)

1. 아래 "환경 구성" 명령으로 `.venv` 생성 (Windows x64 + NVIDIA GPU 필요)
2. 확정 모델 재현 확인. 학습하지 않고 저장된 가중치로 test만 평가:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate.py
```

- 성공 기준: 마지막 줄 `기록된 test accuracy 99.04% → 일치`
- 이후 작업(INT8·RTL)은 직접 학습한 모델이 아니라 `models/mnist_float_v1/best.pt` 기준. GPU 학습은 seed를 고정해도 PC마다 결과가 미세하게 다름

## 학습 환경

- **운영체제:** Windows x64
- **GPU:** NVIDIA GPU 필요 · 현재 검증 장비 RTX 4060 Ti 8GB
- **Python 3.12.15:** 학습 코드 실행
- **PyTorch 2.11.0 · torchvision 0.26.0:** CNN 학습·이미지 데이터 처리 · CUDA 12.8 빌드
- **uv 0.13.0:** Python·라이브러리 설치 및 버전 관리
- **설치 위치:** 프로젝트 내부 `.tools/`, `.venv/`

## 설치 및 검증

### 1. 환경 구성

- 실행 위치: 저장소 폴더
- 실행 도구: PowerShell
- 최초 설치: 인터넷 연결 필요

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

- 실행 내용: uv·Python 설치 → `.venv` 구성 → 라이브러리 설치 → GPU 검증
- 재실행: `uv.lock`에 고정된 버전으로 환경 동기화 후 검증
- 성공 기준: 마지막 출력의 `"status": "PASS"`
- 결과 파일: [docs/validation/environment.json](docs/validation/environment.json)

### 2. 검증만 실행

- `.venv`의 Python으로 GPU 검증 실행

```powershell
.\.venv\Scripts\python.exe .\scripts\check_env.py
```

- CPU·GPU 행렬 연산 결과 비교
- 계획한 CNN 구조의 순전파·역전파·가중치 갱신 확인
- 검증 결과를 터미널에 출력 · 파일 저장 시 `--output <경로>` 추가

## 학습 실행과 진행 상황 확인

- 프로젝트 폴더의 PowerShell 또는 VS Code 터미널에서 실행.
- **아래 명령은 실제 학습을 시작함.** 기본값: 20 epoch, batch 64, Adam 학습률 0.001, seed 42.
- CUDA를 사용할 수 있으면 GPU, 없으면 CPU 선택. 시작할 때 장치 이름 표시.
- 첫 기준 모델은 픽셀을 0~1로 변환하며 데이터 증강은 적용하지 않음.

```powershell
.\.venv\Scripts\python.exe -u .\train.py
```

- 설정을 바꾸는 예시:

```powershell
.\.venv\Scripts\python.exe -u .\train.py --epochs 20 --batch-size 64 --lr 0.001 --log-interval 100
```

- 정확도 개선 옵션(기본은 모두 꺼짐 → 기준 실험과 동일):

| 옵션 | 내용 |
|---|---|
| `--augment` | 학습용만 회전 ±10도·이동 ±10%·크기 90~110% 증강. 검증·test는 원본 |
| `--scheduler cosine` | epoch마다 학습률을 줄여 마지막 epoch에서 0에 가깝게 만듦 |
| `--weight-decay <값>` | 0보다 크면 AdamW로 가중치 감쇠 적용 |

```powershell
.\.venv\Scripts\python.exe -u .\train.py --epochs 30 --augment --scheduler cosine --weight-decay 1e-3
```

- 옵션 설명만 확인하기(학습 시작 안 함):

```powershell
.\.venv\Scripts\python.exe .\train.py --help
```

### 화면에서 확인할 내용

- **배치 진행:** 첫 배치·100배치마다·마지막 배치의 누적 loss와 accuracy 표시.
- **epoch 요약:** 학습/검증 loss·accuracy, epoch 소요 시간, 남은 학습·검증 예상 시간.
- **가중치 저장:** 검증 정확도가 최고일 때 `best.pt` 저장. 정확도가 같으면 검증 손실이 더 낮은 모델 선택.
- **최종 평가:** 모든 epoch가 끝나면 최고 모델을 다시 불러와 공식 test 10,000장 평가. test는 모델 선택에 사용하지 않음.
- **지표 의미:** 화면 정확도는 %, CSV/JSON 정확도는 0~1. 학습 지표는 가중치가 갱신되는 동안의 집계이고, 검증 지표는 해당 epoch 종료 모델의 평가.
- **예상 시간:** 이미 완료된 epoch의 실측 시간을 기준으로 갱신. 최종 test 시간은 제외.

### 저장되는 파일

- 위치: `artifacts/mnist/<실행 시각>/` — 시각은 한국 시간이며, 실행마다 새 폴더 생성.
- 실제 절대 경로는 실행 시작 시 터미널에 표시.

| 파일 | 내용 |
|---|---|
| `best.pt` | 최고 모델의 float 가중치, 선택된 epoch, 검증 지표, 실행 설정 |
| `config.json` | epoch·batch·학습률·optimizer·weight decay·scheduler·증강·seed·모델·장치·PyTorch 버전 등 |
| `history.csv` | 완료된 epoch별 학습률·학습/검증 지표·소요 시간·최고 모델 여부 |
| `train.log` | 터미널에 표시한 진행 상황과 결과·오류 |
| `results.json` | 학습 종료 후 최고 모델의 공식 test 결과 |

- **중단:** `Ctrl+C`. 이미 저장한 최고 모델과 완료된 epoch 기록은 유지. 첫 epoch 완료 전에는 가중치 파일이 없을 수 있음.
- **재실행:** 처음부터 새로 학습하며 이전 결과 폴더는 보존. 중단 지점 자동 재개는 구현하지 않음.
- **주의:** `best.pt`는 PyTorch 실수형 모델용이며 FPGA용 INT8 가중치 변환은 다음 단계.
- 다른 코드에서 가중치 불러오기:

```python
import torch
from train import MnistCNN

checkpoint = torch.load("models/mnist_float_v1/best.pt", map_location="cpu", weights_only=True)
model = MnistCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## 주요 파일·폴더

- `.python-version`: 사용할 Python 버전
- `pyproject.toml`: 필요한 라이브러리·설치 출처
- `uv.lock`: 전체 의존성의 고정 버전
- `.venv/`: 프로젝트 전용 Python 가상환경 · Git 제외
- `.tools/`, `.cache/`: 설치 도구·다운로드 캐시 · Git 제외
- `scripts/setup.ps1`: 환경 설치·동기화·검증
- `scripts/check_env.py`: GPU 동작 검증
- `train.py`: CNN·데이터 준비·학습/검증·기록·최고 모델 저장·최종 test 평가
- `data/`: 데이터셋 저장 위치 · Git 제외
- `artifacts/`: 실행별 학습 결과(실험용) · Git 제외
- `models/`: 확정 모델 가중치·설정·학습 기록 · Git 포함
- `scripts/evaluate.py`: 저장된 모델의 test 평가·기록 일치 확인
- `scripts/analyze_sparsity.py`: validation 기준 층별 0 비율·예상 cycle 분석 → `docs/analysis/`

## 진행 상태

- **P0 완료:** 학습 환경 구성·GPU 연산 검증
- **P1 완료:** float 모델 test 99.04% (`models/mnist_float_v1`), validation 희소성 분석. 상세는 [개발기록.md](개발기록.md)
- **후속 순서:** INT8 양자화 → 정수 기준 모델 → RTL 구현

## 참고

- [uv의 PyTorch 설정](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)

