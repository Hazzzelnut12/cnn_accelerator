# CNN accelerator

Zynq-7020 보드 카메라로 **손글씨 숫자열을 실시간 판독**. LeNet 계열 CNN을 직접 작성한 Verilog로 FPGA(PL)에서 가속.

## 문서 (docs/ 폴더, 번호 순서대로)

| | 문서 | 내용 |
|---|---|---|
| 1 | [개발기록](docs/1_개발기록.md) | 지금까지 한 일과 그 이유 — **먼저 읽을 것** |
| 2 | [계획](docs/2_계획.md) | 목표, 남은 로드맵, 미확정 항목 |
| 3 | [가속기 구조](docs/3_가속기구조.md) | 블록별 역할, 메모리, 레지스터 맵 ([구조도](docs/3_가속기구조.svg)) |
| 4 | [정수 규격](docs/4_정수규격.md) | RTL이 비트 단위로 맞출 계산 규칙 |
| 5 | [논문 조사](docs/5_논문조사.md) | 보고서 관련 연구용 |

## 진행 현황

> **노선 변경 (10/14):** 처음 목표는 화면 중앙 숫자 1개 분류. 그런데 숫자 1개 추론은 0.36 ms로 카메라 프레임(33 ms)의 1%라 가속 효과가 보이지 않음 → 숫자 N개를 읽는 숫자열 판독으로 확장. 숫자 1개 분류는 최소 목표로 유지.

| 단계 | 한 일 | 왜 | 결과 |
|---|---|---|---|
| P0 ✅ | 프로젝트 전용 Python·CUDA 환경 | 누구 PC에서든 같은 결과 | GPU 검증 PASS |
| P1 ✅ | float CNN 학습, 0 비율 분석 | 하드웨어가 처리할 모델 확정, 최적화 근거 확보 | test 99.04% |
| P2 ✅ | INT8 양자화, 정수 전용 reference | FPGA는 정수로 계산. RTL이 맞출 "정답" 필요 | 손실 0.00%p |
| P3 ✅ | 계산 규칙 확정, 층별 정답값(golden) export | RTL이 틀렸을 때 어느 층·좌표인지 바로 찾기 | 재계산 불일치 0 |
| P4a ✅ | 가속기 블록 구조 설계 | Verilog 작성 전에 구조·메모리·인터페이스 확정 | 예상 0.385 ms/crop |
| P4b~ ⏳ | RTL 작성 → 보드 결합 → 숫자열 판독 → 측정 | 보드 도착(10/19~25 예상) 후 진행 | 발표 12월 초 가정 |

## 용어

| 용어 | 뜻 |
|---|---|
| PS / PL | Zynq 칩의 ARM CPU 쪽 / FPGA 회로 쪽 |
| MAC · PE | 곱하고 더하기 1회 · 그걸 하는 회로 한 칸. 우리 설계는 PE 8개 |
| PTQ | 학습이 끝난 float 모델을 재학습 없이 정수로 변환 |
| 재양자화 | 24 bit 누산값을 다음 층이 쓰는 8 bit로 줄이는 연산 (×M, 반올림, shift) |
| golden trace | 정수 reference가 낸 층별 정답값. RTL 출력과 자동 비교 |
| CCL | 이진 영상에서 붙어 있는 잉크 덩어리마다 번호·박스를 붙이는 알고리즘 = 숫자 찾기 |
| crop · N_max | 숫자 하나를 잘라 28×28로 맞춘 이미지 · 한 프레임에 처리할 최대 숫자 수 (후보 16) |
| zero-skip | 입력이 0인 곱셈을 건너뛰어 cycle을 줄이는 최적화 |
| BRAM | FPGA 안의 작은 메모리. weight·중간 결과를 전부 여기에 둠 |

## 실행

필요 조건: Windows x64, 최초 설치 시 인터넷. NVIDIA GPU가 없으면 `setup.ps1`의 마지막 GPU 검증만 실패하고 나머지는 CPU로 동작

```powershell
# 1. 환경 구성. 성공: 마지막 출력 "status": "PASS"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1

# 2. 확정 모델 확인 (학습 없이 test 평가). 성공: "→ 일치"
.\.venv\Scripts\python.exe .\python\evaluate.py
```

| `python\` 파일 | 하는 일 | 결과 위치 |
|---|---|---|
| `train.py --epochs 30 --augment --scheduler cosine --weight-decay 1e-3` | 최종 모델과 같은 설정으로 학습 | `artifacts/mnist/<시각>/` |
| `evaluate.py` | 확정 모델 test 평가·기록 비교 | 화면 |
| `analyze_sparsity.py` | 층별 0 비율·예상 cycle | `results/` |
| `quantize.py --save signed:max` | INT8 설정 4개 비교, 선택 설정 저장 | `models/mnist_int8_v1/`, `results/` |
| `export_golden.py` | 논리 순서 hex + golden trace + 재계산 검증 | `models/mnist_int8_v1/mem/`, `build/golden/` |
| `pack_mem.py` | BRAM 배치 hex·RTL 상수 | `rtl/mem/`, `rtl/include/` |

- **INT8 이후 재생성 순서:** `quantize.py --save signed:max` → `export_golden.py` → `pack_mem.py`
- **모델 공유:** 학습 결과는 `artifacts/`(Git 제외)에 쌓이고 확정한 것만 `models/`로 commit. GPU 학습은 seed를 고정해도 PC마다 미세하게 달라서 가중치 파일 자체를 공유

## 폴더

```text
├─ docs/        사람이 읽는 문서 (위 표)
├─ python/      학습·양자화·export 코드
├─ models/      확정 모델 (float v1, INT8 v1)
├─ rtl/         하드웨어 — include/ 상수, mem/ BRAM 초기값 (자동 생성)
├─ results/     측정 결과 원본 JSON (문서 수치의 출처)
├─ scripts/     환경 설치·GPU 검증
└─ Git 제외:   .venv/  data/  artifacts/ (학습 실행 결과)  build/ (golden trace)
```
