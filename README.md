# 손글씨 숫자열 인식 FPGA

목표는 카메라 한 장면의 여러 숫자를 각각 인식하는 것이다.

최대 16개·여러 줄을 지원한다. MNIST 합성셋으로 숫자별 인식률을 평가하고, 실제 카메라는 동작 시연으로 확인한다.

현재 완료한 작업은 MNIST 숫자 분류 모델의 학습·평가와 프로젝트 평가 기준 합의다.

다음은 조정용 MNIST 합성셋을 만들고, 숫자 분리 과정에 기존 모델을 연결해 16개 입력의 초기 성능을 확인하는 작업이다.

문서는 [프로젝트 계획·진행 기록](docs/계획.md)으로 통합 관리한다. 위쪽 체크리스트로 진행 상태를 확인하고, 아래쪽에 단계별 기록을 남긴다.

모델은 Conv 1→8→16, FC 784→32→10으로 구성했다. 최종 학습은 30 epoch, 데이터 증강, cosine 학습률, AdamW를 사용했다.

| 항목 | 결과 |
|---|---|
| 채택한 모델 | epoch 27 |
| validation 정확도 | 98.98% |
| test 정확도 | 99.04% (9,904 / 10,000) |

## 파일 위치

```text
python/
  train.py              모델 구조와 학습
  evaluate.py           저장된 모델 평가
models/mnist_float_v1/
  best.pt               채택한 가중치
  config.json           학습 설정
  history.csv           epoch별 지표
  results.json          평가 결과
  train.log             학습 로그
build/
  training/             학습 실행 3회의 원본, 새 학습 결과
  environment.json      GPU 환경 확인 결과
data/MNIST/             데이터셋
scripts/                환경 설치와 GPU 확인
```

## 실행

프로젝트 루트의 PowerShell에서 실행한다.

```powershell
# 저장된 모델 평가
.\.venv\Scripts\python.exe .\python\evaluate.py

# 같은 설정으로 새로 학습
.\.venv\Scripts\python.exe .\python\train.py --epochs 30 --augment --scheduler cosine --weight-decay 1e-3
```

새 학습 결과는 `build/training/<실행시각>/`에 저장한다. `models/`의 기존 모델은 그대로 유지한다.

환경을 처음 설치할 때는 다음 명령을 실행한다. 마지막 GPU 확인에는 NVIDIA GPU가 필요하다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

`.tools/`와 `.venv/`는 프로젝트의 Python 실행 환경이다. 패키지 설정은 `pyproject.toml`, 버전 잠금은 `uv.lock`에 있다.
