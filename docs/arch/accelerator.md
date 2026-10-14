# 가속기 구조 (P4a)

구조도 [accelerator.svg](accelerator.svg) · 정수 규격 [int8-spec](../spec/int8-spec.md) · 블록별 상세 설명은 테스트벤치 페이지의 "가속기 구조" 탭

## 1. 결정 사항

| 항목 | 결정 | 근거 |
|---|---|---|
| 엔진 구성 | MAC 배열 하나를 모든 층이 순서대로 공유 | 자원 최소, 병렬도 실험이 파라미터 하나 |
| FC 처리 | Conv와 같은 MAC 배열 | 입력 1개를 PE 8개에 방송하는 패턴이 같고 주소만 다름 |
| Pool | Conv와 결합 (2×2 네 점을 연달아 계산하며 max) | Pool 결과만 저장, Conv1 버퍼 6,272 B → 1,568 B |
| 병렬도 | P = 8, 출력 채널 병렬 | 모든 층에서 PE가 놀지 않음 (Conv1 8채널) |
| 재양자화기 | 1개, 순차 | 출력 8개 처리(8 cycle)가 다음 계산(Conv1 9, Conv2 72 cycle)에 가려짐 |
| 입력 버퍼 | double buffer | 계산 중 다음 crop 쓰기 |
| Weight | ROM, `.mem`으로 합성 시 고정 | 구조 단순, 전원 켜면 바로 사용 |
| ARM 연결 | AXI-Lite 레지스터 + AXI BRAM | crop 784 B는 DMA 없이 충분 |
| 출력 버퍼 | N_max(16)장 분량 | batch 끝나고 한 번에 읽음, 인터럽트 batch당 1회 |
| argmax | 하드웨어에서도 계산 | 비교기 하나, ARM은 예측 숫자만 읽으면 됨 |
| 언어 | RTL Verilog-2001, testbench SystemVerilog | 합성 회로는 Verilog, 검증 코드는 파일 읽기·비교가 간결한 SV |
| 검증 | golden 자동 비교 + 핵심 파형 캡처 | PASS/FAIL 판정과 보고서 근거 |

## 2. 계산 순서 (dense 기준선)

```text
Conv (Pool 결합)
for 풀링 출력 위치 (py, px)        Conv1 14×14, Conv2 7×7
  for 출력 채널 묶음 g             Conv1 1개, Conv2 2개 (8채널씩)
    for 2×2 안의 위치 (dy, dx)     4개
      for 항 t = (c, ky, kx)       Conv1 9개, Conv2 72개      ← 1 cycle에 1항
        PE[i] += x[t] × w[8g+i][t]
      → bias → ReLU → 재양자화 → 채널별 max 갱신 (8개 순차, 다음 위치 계산과 겹침)
    → max 8개를 다음 층 버퍼에 저장

FC
for 출력 묶음 g                    FC1 4개, FC2 2개
  for 항 t = 입력 index            FC1 784개, FC2 32개
    PE[i] += x[t] × w[8g+i][t]
  → FC1: bias → ReLU → 재양자화 → 저장 / FC2: bias → logit 저장 + argmax
```

- **padding:** 경계 밖 항은 메모리를 읽지 않고 x = 0 공급. 기준선은 padding 항도 1 cycle 사용 — 근거: 제어 단순. zero-skip 비교 때 padding 몫은 따로 보고
- **누산 초기화:** 첫 항에 `first` 표시 → acc = x·w로 시작, 마지막 항 `last` 표시 → 결과를 후처리로 넘김 — 근거: 출력 점 사이에 빈 cycle 없음

## 3. 예상 cycle (P = 8, 100 MHz)

| 층 | 계산 | cycle |
|---|---|---:|
| Conv1 | 196 위치 × 1 묶음 × 4 × 9항 | 7,056 |
| Conv2 | 49 × 2 × 4 × 72 | 28,224 |
| FC1 | 4 묶음 × 784 | 3,136 |
| FC2 | 2 × 32 | 64 |
| **합계** | + 층 전환 수십 cycle | **약 38,500 = 0.385 ms** |

- crop 16개: 약 6.2 ms → 카메라 프레임 33 ms 안에 들어감
- P1 분석의 35,524 cycle은 padding을 뺀 값

## 4. 메모리

| 메모리 | 내용 | 크기 | 구현 |
|---|---|---|---|
| 입력 버퍼 A/B | crop 28×28 | 784 B × 2 | BRAM36 1개. ARM 32bit 쓰기, 코어 byte 읽기 |
| Pool1 / Pool2 | 층 사이 결과 | 1,568 B / 784 B | BRAM18 각 1개 |
| FC1 출력 | 32개 | 32 B | 레지스터 |
| Weight ROM | 3,353 word × 64 bit | 26.8 KB | BRAM36 7개 |
| Bias ROM | 66 × INT32 (실제 최대 14 bit) | 264 B | LUT ROM |
| 출력 버퍼 | 16 crop × 16 word × 32 bit | 1 KB | BRAM18 1개, ARM 읽기 |

- **Weight word 배치:** `word[g][t] = {w[8g+7][t], …, w[8g][t]}`, 하위 byte가 PE0. 층 순서 Conv1(0~8) → Conv2(9~152) → FC1(153~3,288) → FC2(3,289~3,352). FC2 두 번째 묶음은 출력 2개만 유효, 나머지 0
- **합계:** BRAM36 약 10개 / 140, DSP 약 9개 / 220 (PE 8 + 재양자화 1)

## 5. 레지스터 맵 (AXI-Lite, 32 bit)

| 주소 | 이름 | 내용 |
|---|---|---|
| 0x00 | CTRL | bit0 start (자동 해제), bit1 soft reset |
| 0x04 | STATUS | bit0 busy, bit1 done (1을 써서 해제), bit2 error (N이 범위 밖) |
| 0x08 | N | 처리할 crop 수 (1~16) |
| 0x0C | IN_VALID | bit0 버퍼 A 채움, bit1 B 채움. ARM이 1을 쓰고, 코어가 다 읽으면 0으로 |
| 0x10 | PROGRESS | 완료한 crop 수 |
| 0x14 | CYCLES | 마지막 batch에 걸린 cycle 수 |
| 0x18 | INFO | P, N_MAX (읽기 전용) |

**출력 버퍼:** crop k → word `16k + 0~9` = logit 0~9 (INT32), `16k + 10` = 예측 숫자

**ARM 사용 순서**
1. crop 0을 A, crop 1을 B에 쓰고 IN_VALID = 0b11, N 설정, start
2. 코어는 A → B → A … 순서로 사용하고 다 읽은 버퍼의 valid를 0으로 → ARM은 빈 버퍼에 다음 crop 쓰기
3. done 인터럽트 → 출력 버퍼에서 예측 숫자 읽기 (신뢰도가 필요하면 logit도)

## 6. Pipeline (Conv1 출력 한 점, 9항)

```text
cycle     0   1   2   3   4   5   6   7   8   9  10  11  12
주소      t0  t1  t2  t3  t4  t5  t6  t7  t8  t0' t1' …           ← 다음 점이 바로 이어짐
BRAM 읽기     t0  t1  …                       t8
곱셈              t0  …                           t8
누산                  t0(first) …                     t8(last)
후처리                                                    ch0 … ch7 (8 cycle)
```

- 단계 수(주소 → 읽기 → 곱셈 → 누산)는 초안. P4b에서 100 MHz timing을 보고 조정

## 7. 파일과 다음 단계

- **메모리 초기화 파일:** [pack_mem.py](../../scripts/pack_mem.py) → `rtl/mem/weights.mem` (64 bit word), `rtl/mem/bias.mem`, `rtl/include/cnn_params.vh` (M·n·시작 주소). 풀어서 원본과 일치 확인 PASS
- **P4b 모듈 작성 순서:** PE → 재양자화·후처리 → 주소 생성기 → 메모리 → 제어 FSM → AXI·top
