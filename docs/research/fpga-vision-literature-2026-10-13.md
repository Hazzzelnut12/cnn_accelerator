# FPGA 영상 신경망 관련 논문 탐색 — 2026-10-13

- 용도: 보고서 관련 연구 자료. 끝부분의 "차별점 후보"는 조사 당시 제안이며, 현재 노선은 [계획](../../계획.md) 기준.

## 조사 범위와 읽는 방법

- **목적:** XC7Z020 기반 자체 CNN 가속기의 차별점을 정하기 위한 선행연구 탐색.
- **범위:** 2024–2026년 연구를 중심으로 관련 논문 **50편**을 선별. 이전 핵심 연구 5편 포함.
- **확인 수준:** 47편은 초록·저자 공개본·공식 출판 자료를 중심으로 내용 확인. 나머지 3편은 공식 학회 프로그램의 제목·게재만 확인.
- **본문 추가 확인:** FlexViT, HASS, HGQ-LUT, ExSpike 등은 설계·평가 부분까지 확인. 모든 논문을 전문 정독하거나 결과를 재현한 조사는 아님.
- **출처:** CVF Open Access, 공식 학회 프로그램, 출판사, 저자 arXiv·기관 저장소·코드. 접근이 제한된 CVF 페이지 일부는 검색에 공개된 공식 초록과 저자 자료로 확인.
- **분류:** CVPR 본회의, CVPR 워크숍, ICCV/WACV, FPGA/FCCM/FPL, 회로·시스템 학회, 저널, 출판처 미확인 공개본을 구분.
- **주의:** 아래 50편이 모두 “CVPR에 실린 FPGA 구현 논문”은 아님. FPGA 논문만 보면 모델 효율화 흐름을 놓치고, CVPR만 보면 회로 구현 선행연구를 놓치므로 양쪽을 연결한 목록.
- **중복 처리:** arXiv와 학회판은 한 편으로 집계. 온라인 공개 연도와 저널 권호 연도가 다른 경우 함께 표기.
- **조사 한계:** 체계적 전수조사·인용 수 순위가 아님. 주제별 대표 사례를 넓게 찾은 목록이며 신규성 확정에는 더 좁은 설계별 선행연구 비교가 필요.

## 먼저 알아둘 용어

| 용어 | 의미 |
|---|---|
| 추론 | 학습이 끝난 가중치로 새 입력의 결과를 계산하는 것. 여기서 대부분의 FPGA 연구가 가속하는 단계 |
| 양자화 | 실수 가중치·중간값을 적은 비트로 표현하는 것 |
| W4A8 | 가중치 4비트, activation 8비트. 누산·스케일 처리 비트 수는 별도 |
| activation | 레이어를 통과하며 만들어지는 중간 결과값 |
| 희소성 | 값 중 0인 항목이 많다는 성질. 일부 논문은 비트·토큰·중요도의 희소성을 뜻하므로 정의 확인 필요 |
| zero-skipping | 0과 관련된 연산을 실제 실행 순서에서 생략하는 것 |
| dataflow | 가중치·입력·중간 결과를 어디에 두고 어떤 순서로 공급하는지 |
| LUT | FPGA의 작은 논리 함수 구현 자원. 논문에서 말하는 큰 lookup table은 여러 물리 LUT로 구현될 수 있음 |
| PS / PL | Zynq의 ARM 프로세서 영역 / 프로그래머블 로직 영역 |
| end-to-end | 측정 시작점부터 결과까지의 전체 경로. 논문마다 범위가 다르므로 센서·전송·CPU 포함 여부 확인 |

## 조사에서 반복해서 나타난 특징

### 1. 양자화와 회로를 함께 설계한다

“학습한 모델을 INT8로 변환”하는 수준에서 더 나아가, 값의 분포와 실제 회로 비용을 함께 본다.

- DASQ: 가중치의 희소 이상치와 대칭 성분을 나눠 zero-point 보정 비용을 줄이는 방향.
- AHCPTQ: 작은 값과 이상치를 다르게 표현하고 곱셈·시프트 경로를 설계.
- HGQ: 필요한 정밀도를 더 세밀하게 학습.
- 우리에게 주는 질문: 모든 레이어가 8비트여야 하는가? 비트를 줄였을 때 저장 공간만 줄어드는가, 실제 연산 병렬성도 늘어나는가?

출처: [DASQ](https://openaccess.thecvf.com/content/CVPR2026/html/Mori_Rethinking_Asymmetric_Quantization_Hidden_Symmetry_in_Vision_Model_Weights_CVPR_2026_paper.html), [AHCPTQ](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_AHCPTQ_Accurate_and_Hardware-Compatible_Post-Training_Quantization_for_Segment_Anything_Model_ICCV_2025_paper.pdf), [HGQ](https://authors.library.caltech.edu/records/xypmg-afb90).

### 2. MAC 수보다 데이터 이동과 병목을 함께 본다

곱셈기를 늘려도 데이터 공급이 느리거나 중간 결과를 자주 저장하면 전체 성능은 개선되지 않는다.

- PipeViT: attention 연산 융합·타일링·파이프라인으로 메모리 부담 감소.
- FlexViT: Conv·FC에 공유 GEMM을 쓰고 온칩 타일에 부분합을 유지.
- SpiderCam: 영상 처리 알고리즘과 스트리밍 메모리 사용을 함께 설계.
- 우리에게 주는 질문: Conv → ReLU → Pool 사이에 무엇을 저장해야 하며, 버퍼 읽기·쓰기를 얼마나 줄일 수 있는가?

출처: [PipeViT 공식 초록](https://www.aspdac.com/aspdac2026/program/program-abstract.html), [FlexViT](https://arxiv.org/abs/2606.31938), [SpiderCam](https://openaccess.thecvf.com/content/CVPR2026/html/Ferreira_SpiderCam_Low-Power_Snapshot_Depth_from_Differential_Defocus_CVPR_2026_paper.html).

### 3. 희소성은 “0의 비율”보다 처리 구조가 중요하다

- NullHop부터 activation의 0을 건너뛰는 연구가 존재한다. zero-skipping 자체를 새로운 아이디어라고 주장하기는 어렵다.
- HASS는 가중치·activation 희소성에 더해 레이어별 하드웨어 비용과 처리량을 고려한다.
- ESDA는 이벤트의 희소성을 유지하고 레이어 사이에 전달하는 구조를 설계한다.
- ExSpike는 이벤트 압축·주소 처리까지 고려한다. 다만 SNN이므로 일반 CNN에 그대로 적용하는 기술은 아니다.
- 우리에게 주는 질문: 0을 검사하고 인덱스를 관리하는 비용보다 생략한 연산의 이득이 큰가? 희소하지 않은 입력에서도 손해가 제한되는가?

출처: [NullHop](https://arxiv.org/abs/1706.01406), [HASS](https://arxiv.org/abs/2406.03088), [ESDA](https://doi.org/10.1145/3626202.3637558), [ExSpike](https://arxiv.org/abs/2606.20414).

### 4. FPGA의 LUT를 적극적으로 활용한다

- TLMAC·LUTMUL: 저비트 곱셈·MAC을 soft logic에 매핑.
- NeuraLUT 계열·HGQ-LUT·KANELÉ: 신경망 함수 자체를 논리·lookup 구조에 맞춰 설계.
- 공통 질문: DSP 사용을 줄이는 대신 LUT·배선·학습 복잡도가 얼마나 늘어나는가?
- 작은 lookup도 입력 비트 수가 증가하면 경우의 수가 급격히 늘어난다. “표로 바꾸면 항상 싸다”는 뜻은 아니다.

출처: [TLMAC](https://arxiv.org/abs/2403.11414), [LUTMUL](https://arxiv.org/abs/2411.11852), [HGQ-LUT](https://arxiv.org/abs/2604.22293), [KANELÉ](https://arxiv.org/abs/2512.12850).

### 5. 모델 종류가 다양해지면서 행렬곱 이외의 연산도 다룬다

ViT·Vision Mamba·Vision GNN·MoE가 연구 대상으로 등장한다. 각각 Softmax/정규화, 상태 갱신, 그래프 생성, 전문가 선택 등 다른 병목을 만든다. 따라서 큰 모델을 선택하는 것만으로 가속기의 기여가 생기는 것은 아니다.

출처: [Refining Datapath for Microscaling ViTs](https://arxiv.org/abs/2505.22194), [ViM-Q](https://arxiv.org/abs/2605.01935), [GraphLeap](https://arxiv.org/abs/2604.21290), [UbiMoE](https://arxiv.org/abs/2502.05602).

### 6. 코어 성능과 실제 시스템 효과를 구분해야 한다

FlexViT는 우리와 같은 XC7Z020 계열인 PYNQ-Z2를 사용한다. 저자 표에서는 가속 대상 레이어가 최대 2.74배 빨라져도 전체 추론 향상은 최대 1.40배다. CPU에 남은 처리 때문이며, 일부 모델은 추론 시간이 줄어도 에너지가 늘었다.

이는 모든 FPGA 가속기의 한계를 뜻하는 것이 아니라, **특정 모델·분담·측정 조건에서 나온 결과**다. 우리도 CNN 실행 시간과 전처리·전송·표시까지 포함한 시간의 차이를 측정해야 한다.

출처: [FlexViT 본문 Table V](https://arxiv.org/html/2606.31938v1).

## 논문 50편 목록

각 행의 논문명은 원문·공식 자료 링크다. “적용 시 주의”는 위 자료에 근거한 우리 프로젝트 관점의 해석이다.

### A. CVPR·ICCV에서 FPGA 구현과 직접 연결되는 연구

| 번호 | 논문 | 발표·공개 | 핵심 내용·확인 범위 | 우리에게 주는 시사점 / 적용 시 주의 |
|---:|---|---|---|---|
| 1 | [Rethinking Asymmetric Quantization: Hidden Symmetry in Vision Model Weights (DASQ)](https://openaccess.thecvf.com/content/CVPR2026/html/Mori_Rethinking_Asymmetric_Quantization_Hidden_Symmetry_in_Vision_Model_Weights_CVPR_2026_paper.html) | CVPR 2026 본회의 | 희소한 이상치와 나머지 대칭 분포를 분리해 zero-point 보정 비용을 줄인다. FPGA 평가 포함. | 양자화 방식을 고를 때 정확도와 곱셈·보정 회로를 함께 고려. activation zero-skip과는 다른 접근. |
| 2 | [SpiderCam: Low-Power Snapshot Depth from Differential Defocus](https://openaccess.thecvf.com/content/CVPR2026/html/Ferreira_SpiderCam_Low-Power_Snapshot_Depth_from_Differential_Defocus_CVPR_2026_paper.html) | CVPR 2026 본회의 | 차등 디포커스 기반 깊이 추정을 SystemVerilog와 메모리 효율적인 영상 처리로 구현. 신경망 가속 논문은 아님. | 모델 크기보다 센서·알고리즘·회로를 함께 설계한 시스템 기여를 참고. |
| 3 | [Towards Low-Latency Event-based Obstacle Avoidance on a FPGA-Drone](https://openaccess.thecvf.com/content/CVPR2025W/EventVision/html/Bonazzi_Towards_Low-Latency_Event-based_Obstacle_Avoidance_on_a_FPGA-Drone_CVPRW_2025_paper.html) | CVPR 2025 워크숍 · EventVision | 이벤트 카메라와 FPGA 추론을 연결한 드론 장애물 회피. 센서 집계와 추론을 포함한 지연 평가. | CNN 코어 시간과 센서부터 결과까지의 지연을 따로 측정. |
| 4 | [Live Demonstration: Real-time event-data processing with Graph Convolutional Neural Networks and SoC FPGA](https://openaccess.thecvf.com/content/CVPR2025W/EventVision/html/Wzorek_Live_Demonstration_Real-time_event-data_processing_with_Graph_Convolutional_Neural_Networks_CVPRW_2025_paper.html) | CVPR 2025 워크숍 · EventVision · 데모 | 이벤트 데이터를 그래프로 구성하고 GCN 특징을 FPGA에서 처리. PS와 PL이 작업을 분담. | 전체 모델이 PL에 들어간 구현과 구분해야 함. 짧은 데모 논문. |
| 5 | [Co-designing a Sub-millisecond Latency Event-based Eye Tracking System with Submanifold Sparse CNN](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Zhang_Co-designing_a_Sub-millisecond_Latency_Event-based_Eye_Tracking_System_with_Submanifold_CVPRW_2024_paper.html) | CVPR 2024 워크숍 · AI4Streaming | 이벤트 입력의 희소성을 유지하는 CNN과 FPGA를 공동 설계. GRU·FC 등 CPU 처리 부분 존재. | 일반 프레임 카메라와 입력 특성이 다름. 희소성 보존 방식은 참고 가능. |
| 6 | [AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_AHCPTQ_Accurate_and_Hardware-Compatible_Post-Training_Quantization_for_Segment_Anything_Model_ICCV_2025_paper.pdf) | ICCV 2025 본회의 | 로그·균등 혼합 양자화와 채널 그룹화. FPGA에 곱셈과 시프트 처리 경로 구현. | 하드웨어 친화 양자화의 사례. 보충자료상 FP32 처리 구간도 있어 완전 정수 파이프라인으로 해석하면 안 됨. |

### B. CVPR·ICCV·WACV의 모델 효율화 연구 — FPGA 실측과 구분

| 번호 | 논문 | 발표·공개 | 핵심 내용·확인 범위 | 우리에게 주는 시사점 / 적용 시 주의 |
|---:|---|---|---|---|
| 7 | [LS-ViT: Least-Squares Hessian Based Block Reconstruction for Low-Bit Post-Training Quantization](https://openaccess.thecvf.com/content/CVPR2026/html/Hwang_LS-ViT_Least-Squares_Hessian_Based_Block_Reconstruction_for_Low-Bit_Post-Training_Quantization_CVPR_2026_paper.html) | CVPR 2026 본회의 | 저비트 PTQ에서 블록 재구성의 오차와 계산 비용을 개선. | 저비트 정확도 복구 아이디어. FPGA 속도 향상의 직접 증거로 사용하지 않음. |
| 8 | [FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_FIMA-Q_Post-Training_Quantization_for_Vision_Transformers_by_Fisher_Information_Matrix_CVPR_2025_paper.html) | CVPR 2025 본회의 | Fisher 정보 근사로 양자화 후 재구성 오차를 다룬다. | 정확도 손실을 줄이는 방법과 실제 하드웨어 매핑은 별도 검증 대상. |
| 9 | [APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_APHQ-ViT_Post-Training_Quantization_with_Average_Perturbation_Hessian_Based_Reconstruction_for_CVPR_2025_paper.html) | CVPR 2025 본회의 | Hessian 기반 블록 재구성으로 ViT PTQ를 개선. | 학습 전체를 반복하지 않고 저비트 정확도를 복구하는 계열. |
| 10 | [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Q-DiT_Accurate_Post-Training_Quantization_for_Diffusion_Transformers_CVPR_2025_paper.html) | CVPR 2025 본회의 | Diffusion Transformer의 그룹 크기와 activation 양자화를 조정. | 시간·입력에 따라 값의 분포가 달라지는 문제. 현재 숫자 CNN에는 우선순위 낮음. |
| 11 | [ICP: Immediate Compensation Pruning for Mid-to-high Sparsity](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_ICP_Immediate_Compensation_Pruning_for_Mid-to-high_Sparsity_CVPR_2025_paper.html) | CVPR 2025 본회의 | 높은 희소도에서 가지치기로 생기는 오차를 보상. | 0의 비율을 높이는 알고리즘이며, 그만큼 FPGA가 빨라지는지는 별도 문제. |
| 12 | [RepViT: Revisiting Mobile CNN From ViT Perspective](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html) | CVPR 2024 본회의 | ViT의 설계 관점을 반영해 모바일 CNN 구조를 개선. | CNN 자체도 여전히 연구 대상. 모바일 지연 결과를 FPGA 결과로 대체하면 안 됨. |
| 13 | [SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design](https://openaccess.thecvf.com/content/CVPR2024/html/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.html) | CVPR 2024 본회의 | 초기 단계와 attention 구조를 조절해 메모리 비용을 줄이는 경량 ViT. | MAC 수 외에 중간 데이터 크기와 이동을 볼 근거. |
| 14 | [MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_MixA-Q_Revisiting_Activation_Sparsity_for_Vision_Transformers_from_a_Mixed-Precision_ICCV_2025_paper.pdf) | ICCV 2025 본회의 | activation의 중요도 차이를 혼합 정밀도로 활용. | 덜 중요한 값을 저비트로 처리하는 것과 정확히 0인 값을 건너뛰는 것은 다름. |
| 15 | [MSQ: Memory-Efficient Bit Sparsification Quantization](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Han_MSQ_Memory-Efficient_Bit_Sparsification_Quantization_ICCV_2025_paper.pdf) | ICCV 2025 본회의 | 비트 희소화와 양자화를 통해 메모리 효율을 개선. | 학습·양자화 과정의 메모리 개선과 FPGA 추론 성능을 구분. |
| 16 | [Hardware Aware Evolutionary Neural Architecture Search Using Representation Similarity Metric](https://openaccess.thecvf.com/content/WACV2024/html/Sinha_Hardware_Aware_Evolutionary_Neural_Architecture_Search_Using_Representation_Similarity_Metric_WACV_2024_paper.html) | WACV 2024 | 표현 유사성과 하드웨어 비용을 이용한 구조 탐색. | 채널 수·연산 구성을 실제 장치 비용과 함께 선택하는 접근. |
| 17 | [Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Gafni_Dual_Precision_Quantization_for_Efficient_and_Accurate_Deep_Neural_Networks_CVPRW_2025_paper.html) | CVPR 2025 워크숍 · eLVM | 가중치 저장은 INT4, 연산은 FP8을 활용하는 이중 정밀도. | 저장 비트 수와 연산 비트 수를 각각 확인해야 함. |
| 18 | [Fixing Quantization with Lightweight Adapters](https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Mohammadi_Fixing_Quantization_with_Lightweight_Adapters_CVPRW_2026_paper.html) | CVPR 2026 워크숍 · ECV | 가벼운 어댑터로 양자화 오류를 보정. | 보정 모듈이 추가하는 하드웨어 비용까지 검토 필요. |
| 19 | [Hardware-Aware Neural Feature Extraction for Resource-Constrained Devices](https://openaccess.thecvf.com/content/CVPR2026W/EVW/html/Tosini_Hardware-Aware_Neural_Feature_Extraction_for_Resource-Constrained_Devices_CVPRW_2026_paper.html) | CVPR 2026 워크숍 · EVW | 메모리·연산 제약을 반영한 특징 추출기 탐색과 증류. MCU 대상. | 자원 제약을 학습·구조 설계에 반영하는 사례이며 FPGA 논문은 아님. |

### C. FPGA 전문 학회·설계 학회·저널의 최근 연구

| 번호 | 논문 | 발표·공개 | 핵심 내용·확인 범위 | 우리에게 주는 시사점 / 적용 시 주의 |
|---:|---|---|---|---|
| 20 | [FlexViT: A Flexible FPGA-based Accelerator for Edge Vision Transformers](https://arxiv.org/abs/2606.31938) | FPL 2026 · 저자 공개본 | PYNQ-Z2에서 INT8 GEMM을 Conv·FC에 공유. 타일링과 이중 방송 모드로 데이터 재사용. | 우리와 같은 XC7Z020 계열. CPU에 남은 연산 때문에 코어 향상과 전체 향상에 큰 차이. |
| 21 | [ExSpike: A General Full-Event Neuromorphic Architecture for Exploiting Irregular Sparsity with Event Compression](https://arxiv.org/abs/2606.20414) | FPL 2026 · 저자 공개본 | SNN의 불규칙한 이벤트를 압축하고 이벤트 중심으로 연산을 진행. | 압축·주소 비용까지 고려하는 사례. 일반 CNN과 다르며 전력은 Vivado 추정. |
| 22 | [HGQ-LUT: Fast LUT-Aware Training and Efficient Architectures for DNN Inference](https://arxiv.org/abs/2604.22293) | FCCM 2026 | LUT-Dense·LUT-Conv와 혼합 정밀도를 학습 과정에 반영하고 RTL로 변환. | 주요 평가는 고에너지 물리 데이터. 보고된 ns 지연을 영상 CNN에 그대로 적용할 수 없음. |
| 23 | [HGQ: High Granularity Quantization for Real-time Neural Networks on FPGAs](https://authors.library.caltech.edu/records/xypmg-afb90) | FPGA 2026 | 세밀한 단위의 비트 폭을 학습해 정확도와 FPGA 자원 비용을 조정. | 층마다 8비트를 고정하는 접근보다 세밀한 공동 설계. 이전 arXiv 버전과 중복 집계하지 않음. |
| 24 | [KANELÉ: Kolmogorov–Arnold Networks for Efficient LUT-based Evaluation](https://arxiv.org/abs/2512.12850) | FPGA 2026 | KAN을 LUT 기반 평가에 맞게 설계·양자화. | 일반 MAC 배열 외의 선택지. MNIST 등 저자 코드 공개, 현재 CNN 구조를 유지하는 변경은 아님. |
| 25 | [ViM-Q: Scalable Algorithm-Hardware Co-Design for Vision Mamba Model Inference on FPGA](https://arxiv.org/abs/2605.01935) | FCCM 2026 | Vision Mamba에 저비트 가중치·activation 양자화와 전용 연산 파이프라인을 결합. | 모델의 특수 연산까지 다루는 공동 설계. ZCU102 대상이라 자원 조건이 다름. |
| 26 | [GraphLeap: Decoupling Graph Construction and Convolution for Vision GNN Acceleration on FPGA](https://arxiv.org/abs/2604.21290) | FCCM 2026 | 그래프 구성과 합성곱의 의존 관계를 줄여 실행을 겹친다. | 행렬곱 외의 그래프 생성도 병목이라는 사례. 현재 CNN에는 직접 적용 어려움. |
| 27 | [PipeViT: Accelerating Vision Transformers via Intra-Layer Pipelining](https://www.aspdac.com/aspdac2026/program/program-abstract.html) | ASP-DAC 2026 | attention 연산 융합·타일링·파이프라인과 재구성 가능한 이종 코어. | 메모리 이동 감소와 연산 중첩을 함께 설계. 공식 초록 확인, 보드 세부는 미확인. |
| 28 | [An FPGA-Based Frequency-Focused Vision Transformer Accelerator for Real-Time Inference on Edge Platforms](https://ieeexplore.ieee.org/document/11165448/) | IEEE TCSI 2026 · 온라인 2025 | SpectFormer의 주파수·attention 연산을 공유하고 데이터 전달과 Softmax를 최적화. | 서로 다른 연산을 같은 자원으로 처리하는 사례. KV260과 우리 보드는 다름. |
| 29 | [An FPGA Accelerator for Vision Transformer with Quantization and LUT-Based Operations](https://globals.ieice.org/en_transactions/information/10.1587/transinf.2025PAP0003/_f) | IEICE Transactions on Information and Systems 2026 | 저비트 양자화와 LUT 기반 선형·비선형 연산을 조합. | DSP뿐 아니라 LUT 자원 활용을 검토할 근거. 초록 중심 확인. |
| 30 | [Refining Datapath for Microscaling ViTs](https://arxiv.org/abs/2505.22194) | FPL 2025 | 공유 스케일을 사용하는 MXInt와 ViT의 비선형 연산 데이터 경로를 설계. | Softmax·LayerNorm까지 포함할 때의 수치 형식과 회로 비용을 다룸. |
| 31 | [LUTMUL: Exceed Conventional FPGA Roofline Limit by LUT-based Efficient Multiplication for Neural Network Inference](https://arxiv.org/abs/2411.11852) | ASP-DAC 2025 | LUT 기반 곱셈과 데이터 흐름으로 DSP 중심 연산 한계를 보완. | LUT·배선·동작 주파수까지 포함해야 실제 이득 판단 가능. |
| 32 | [NeuraLUT-Assemble: Hardware-aware Assembling of Sub-Neural Networks for Efficient LUT Inference](https://arxiv.org/abs/2504.00592) | FCCM 2025 | 작은 하위 신경망의 LUT 구현을 조합해 확장성 문제를 완화. | 전체 함수를 거대한 표 하나로 만드는 방식과 구분. MNIST 등 포함. |
| 33 | [Greater than the sum of its LUTs: scaling up LUT-based neural networks with AmigoLUT](https://spiral.imperial.ac.uk/entities/publication/a1279586-22c1-439c-8a09-eae46948ffd5) | FPGA 2025 | 작은 신경망들을 조합해 LUT 기반 신경망의 확장성을 개선. | 입력 비트가 늘면 lookup table 크기가 급격히 증가하는 문제를 다룸. |
| 34 | [UbiMoE: A Ubiquitous Mixture-of-Experts Vision Transformer Accelerator With Hybrid Computation Pattern on FPGA](https://arxiv.org/abs/2502.05602) | ISCAS 2025 | 스트리밍 attention과 재사용 가능한 선형 연산 엔진을 결합. | 연산 특성에 따라 서로 다른 처리 방식을 섞는 사례. ZCU102·U280 대상. |
| 35 | [CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA](https://arxiv.org/abs/2506.08496) | arXiv 2025 · 이 문서에서는 출판처 확정 보류 | MoE ViT의 양자화와 스트리밍·재사용 연산 배치를 함께 조절. | 양자화 결과가 스케줄과 데이터 이동에 영향을 준다는 사례. |
| 36 | [NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions](https://arxiv.org/abs/2403.00849) | FPL 2024 | 신경망 하위 함수를 합성 가능한 논리로 바꿔 LUT에서 처리. | MAC 기반 가속기와 다른 계열의 설계. 작은 모델에서도 연구 가능. |
| 37 | [Table-Lookup MAC: Scalable Processing of Quantised Neural Networks in FPGA Soft Logic](https://arxiv.org/abs/2403.11414) | FPGA 2024 | 양자화된 MAC을 lookup 기반 soft logic에 매핑하고 중복을 활용. | W4A8 같은 후보의 저장·연산 방식 탐색에 유용. 단순 활성화 함수 LUT와 다름. |
| 38 | [A Composable Dynamic Sparse Dataflow Architecture for Efficient Event-based Vision Processing on FPGA (ESDA)](https://doi.org/10.1145/3626202.3637558) | FPGA 2024 | 희소 토큰·특징 인터페이스로 레이어를 연결하고 submanifold convolution을 사용. | 희소성이 레이어를 지나며 사라지는 문제와 중간 버퍼 설계를 참고. |
| 39 | [HASS: Hardware-Aware Sparsity Search for Dataflow DNN Accelerator](https://arxiv.org/abs/2406.03088) | FPL 2024 | weight·activation 희소성과 FPGA 비용을 함께 탐색해 레이어 처리량을 맞춘다. | U250·16비트 평가. 일부 모델은 속도보다 DSP 절약 효과가 나타남. |
| 40 | [An FPGA-Based Reconfigurable Accelerator for Convolution-Transformer Hybrid EfficientViT](https://arxiv.org/abs/2403.20230) | ISCAS 2024 | Conv와 attention의 연산 자원을 공유하고 융합 실행으로 이동 비용 감소. | 동명 EfficientViT 모델이 여럿이므로 모델 계열 확인 필요. ZCU102 대상. |
| 41 | [An efficient CNN accelerator for pattern-compressed sparse neural networks on FPGA](https://www.sciencedirect.com/science/article/pii/S0925231224014711) | Neurocomputing 2025 · 온라인 2024 | 패턴 기반 희소성, 커널 그룹화, 활성값 접근 충돌 완화를 결합. | 0을 찾아도 메모리 접근이 충돌하면 이득을 잃는다는 점이 우리 설계와 연결. |
| 42 | [RQP: Resource-Oriented Quantiser Pruning for Neural Networks on FPGAs](https://arxiv.org/abs/2606.30382) | arXiv 2026 · 출판처 미확인 | FPGA 자원을 기준으로 quantiser를 가지치기하고 미세 조정. | 정확도·자원 절충점 탐색. 주요 데이터는 입자물리이며 영상 논문으로 분류하지 않음. |

### D. 차별점을 판단하기 위한 이전 핵심 연구

| 번호 | 논문 | 발표·공개 | 핵심 내용·확인 범위 | 우리에게 주는 시사점 / 적용 시 주의 |
|---:|---|---|---|---|
| 43 | [NullHop: A Flexible Convolutional Neural Network Accelerator Based on Sparse Representations of Feature Maps](https://arxiv.org/abs/1706.01406) | arXiv 2017 · TNNLS 온라인 2018 / 권호 2019 | 희소 feature map 표현과 0 건너뛰기를 결합한 CNN 가속기. | zero-skipping 자체는 오래된 아이디어. FPGA 결과와 ASIC 평가 수치를 구분. |
| 44 | [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference](https://arxiv.org/abs/1612.07119) | FPGA 2017 | 이진 신경망을 레이어별 FPGA 데이터 흐름에 매핑. | 저비트·전용 파이프라인의 기준 연구. INT8과 정확도·표현력이 다름. |
| 45 | [Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks](https://research.nvidia.com/publication/2016-06_eyeriss-spatial-architecture-energy-efficient-dataflow-convolutional-neural) | ISCA 2016 · ASIC | row-stationary 데이터 재사용으로 이동 에너지를 줄인다. | FPGA 논문은 아님. dataflow와 에너지 비용을 이해하는 비교 자료. |
| 46 | [DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html) | CVPR 2022 본회의 · GPU | 영상 프레임 간 변화가 적다는 점을 활용해 반복 계산을 줄인다. | 카메라 입력 특성 활용 아이디어. FPGA 실측은 아니며 작은 변화 생략은 오차 관리 필요. |
| 47 | [HAQ: Hardware-Aware Automated Quantization With Mixed Precision](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html) | CVPR 2019 본회의 | 하드웨어 비용 피드백으로 혼합 정밀도를 자동 선택. | 혼합 정밀도 자체의 신규성을 주장하기 어려운 이유. 구체적인 FPGA 구현 기여가 필요. |

### E. 공식 프로그램에서 제목·게재만 확인한 추가 후보

| 번호 | 논문 | 발표·공개 | 핵심 내용·확인 범위 | 우리에게 주는 시사점 / 적용 시 주의 |
|---:|---|---|---|---|
| 48 | [EQViTA: An End-to-End Quantized Vision Transformer Accelerator Implemented on Resource-Constrained FPGAs](https://2025.fpl.org/program/program/) | FPL 2025 | 제목·프로그램 확인. 본문·평가 조건 미확인. | 자원 제한 FPGA와 전체 양자화라는 주제가 가까움. 성능 수치는 인용하지 않음. |
| 49 | [ReCoVLM: A Reconfigurable FPGA–GPU Co-Design for Edge Vision-Language Inference](https://www.fccm.org/past/2026/fccm-26-program/) | FCCM 2026 | 제목·프로그램 확인. 본문·평가 조건 미확인. | FPGA와 GPU 공동 설계 후보. 현재 Zynq 단독 설계와 조건이 다름. |
| 50 | [ImageHD: Energy-Efficient On-Device Continual Learning of Visual Representations via Hyperdimensional Computing](https://www.fccm.org/past/2026/fccm-26-program/) | FCCM 2026 | 제목·프로그램 확인. 본문·평가 조건 미확인. | 일반 CNN 외의 영상 표현·장치 내 학습 연구 후보. |

## 우리 보드에서 우선 읽을 논문

| 우선순위 | 논문 | 본문에서 확인할 부분 |
|---:|---|---|
| 1 | FlexViT | 같은 FPGA 계열의 자원 한계, CPU/PL 분담, 전체 지연과 에너지 |
| 2 | HASS | 희소도와 처리량이 다른 이유, 레이어 균형, DSP 절약과 속도 향상의 구분 |
| 3 | NullHop | 기존 zero-skipping 구조. 우리 설계와 실제로 다른 지점이 무엇인지 |
| 4 | TLMAC | 저비트 MAC을 LUT로 구현하는 방법과 확장 비용 |
| 5 | DASQ | 가중치 분포를 수치 형식·회로 구조와 연결하는 방법 |
| 6 | ESDA | 희소 데이터를 레이어 간 전달하는 인터페이스와 버퍼 |
| 7 | SpiderCam | 작은 장치에서 센서부터 결과까지 완성한 시스템 설계 |
| 8 | HGQ-LUT | 학습 단계부터 FPGA 논리 자원을 고려하는 다른 설계 방향 |

### 비교할 때 특히 주의할 사례

- **HASS:** U250·16비트 모델 평가다. 우리 XC7Z020·INT8과 직접 FPS 순위를 비교할 수 없다. MobileNetV3에서는 LUT/BRAM 한계 때문에 처리량은 비슷하고 DSP 사용량이 줄어드는 결과도 보인다. [본문 VI](https://arxiv.org/html/2406.03088v1)
- **FlexViT:** BRAM 83.9%, DSP 71.8%, LUT 69.3%를 사용한다. 같은 칩 계열이라고 해서 카메라·LCD 회로까지 함께 들어갈 여유가 보장되는 것은 아니다. [본문 V-B](https://arxiv.org/html/2606.31938v1)
- **ExSpike:** XC7V2000T 기반 SNN 평가이며, 전력은 벤치마크 활동 정보를 이용한 Vivado 추정이다. 계측기로 측정한 보드 전력과 구분한다. [본문 평가](https://arxiv.org/html/2606.20414v1)
- **HGQ-LUT:** 주요 벤치마크는 입자 분류·추적이다. 빠른 지연 수치가 곧 카메라 CNN 전체의 지연을 뜻하지 않는다. [본문 V](https://arxiv.org/html/2604.22293v1)
- **AHCPTQ:** FPGA용 곱셈·시프트 엔진을 갖추지만 보충자료에 FP32 구간도 존재한다. “모든 연산이 정수”인 우리 목표와 동일하지 않다. [보충자료](https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Zhang_AHCPTQ_Accurate_and_ICCV_2025_supplemental.pdf)
- **CVPR 이벤트 연구:** 이벤트 카메라는 변화 이벤트를 입력으로 준다. 우리 일반 프레임 카메라를 그대로 연결하면 같은 희소성·지연 조건이 나오지 않는다.
- **코드 공개:** 링크가 있다고 재현이 완성되는 것은 아니다. HASS 저장소는 확인 당시 구현 전체가 아닌 안내 중심이었다.

## 우리 프로젝트의 차별점 후보 — 아직 결정하지 않은 제안

현재 계획의 XC7Z020·직접 작성한 RTL·숫자 CNN을 기준으로 한 해석이다. 연구 결과나 확정 사양이 아니다.

### 후보 A. 입력 희소도에 따라 동작하는 skip 구조

- **질문:** 정확히 0인 activation이 많을 때 연산을 줄이면서, 0이 적을 때의 손해를 제한할 수 있는가?
- **설계 후보:** 여러 activation을 묶은 그룹 검사, 비트마스크, 연속된 0 처리, dense 경로 선택.
- **비교:** 같은 양자화 모델의 dense / 단순 skip / 제안 skip.
- **필요 측정:** 레이어별 사이클, 검사·인덱스 비용, 메모리 접근 수, LUT/BRAM 추가량, 가장 불리한 입력.
- **정확도:** 정확한 0만 생략하고 동일한 누산·반올림을 유지한다면 정수 기준 출력과 일치하도록 설계할 수 있다. 임계값으로 작은 비영 값을 버리면 별도 정확도 평가가 필요.
- **신규성:** 동적 skip·그룹화·dense 전환 자체가 새롭다고 확정할 수 없다. 선택한 구체적 구조와 기존 방식의 차이를 추가 조사해야 한다.

### 후보 B. 레이어별 저비트 연산과 DSP/LUT 분담

- **질문:** 정확도 손실을 제한하면서 일부 가중치를 4비트로 줄여 자원이나 처리시간을 개선할 수 있는가?
- **설계 후보:** W8A8 기준, 일부 레이어 W4A8, 저장 packing, DSP와 LUT 매핑 비교.
- **필요 측정:** 정확도, 실제 구현 후 주파수, 레이어 지연, LUT/DSP/BRAM, 스케일 변환 비용.
- **주의:** 같은 8비트 곱셈기에 4비트 값을 넣기만 하면 처리량이 자동으로 두 배가 되지 않는다.

### 후보 C. Conv → ReLU → Pool의 중간 저장 최소화

- **질문:** 연산 사이에 버퍼 전체를 쓰고 다시 읽는 일을 줄일 수 있는가?
- **설계 후보:** line buffer, 융합 파이프라인, 필요한 행·타일만 보관.
- **필요 측정:** BRAM 용량·포트 충돌·읽기/쓰기 수·전체 지연.
- **주의:** 현재 작은 CNN은 온칩 저장이 가능하므로 DDR 절감 효과를 과장하면 안 된다. BRAM 접근·버퍼 크기·지연에서 실제 이득이 있는지 확인한다.

### 선택 순서

1. 작은 CNN을 학습해 레이어별 activation의 **정확한 0 비율, 0의 공간적 분포, 연속 길이**를 측정.
2. 같은 모델의 정수 기준 구현과 가장 단순한 dense RTL 비용을 산정.
3. A/B/C 중 관측된 병목에 맞는 한 가지를 주제로 선택.
4. 단순 기준 구현과 비교해 실제 이득과 추가 비용을 제시.
5. 일반화 주장이 필요하면 별도 데이터·두 번째 소형 모델로 확인. MNIST의 검은 배경만으로 얻은 결과를 모든 CNN의 이득으로 주장하지 않음.

위 순서는 제안이며, 이 조사에서 학습 코드나 RTL 구현을 시작하지 않았다.

## 검증 결과를 어떻게 제시할 것인가

| 검증 항목 | 기록할 내용 | 놓치기 쉬운 점 |
|---|---|---|
| 정확도 | float, quantized, 하드웨어 출력 기준 정확도 | 데이터·전처리·모델이 다르면 단순 비교 불가 |
| 수치 일치 | 정수 reference와 레이어별 RTL 출력 비교 | 누산 순서·overflow·반올림·포화 조건 |
| 코어 성능 | 레이어별 cycle, batch 1 지연, 실제 달성 주파수 | 목표 주파수와 구현 후 달성값 구분 |
| 전체 지연 | 입력 전달, 전처리, CNN, 결과 처리의 범위 | 코어 가속 배율과 카메라 응답 속도를 혼동하지 않음 |
| 자원 | DSP/LUT/FF/BRAM, 기준 구현 대비 변화 | DSP 감소가 LUT 증가로 바뀔 수 있음 |
| 희소성 | 레이어별 0 비율·분포, 실제 생략 연산·사이클 | 80%가 0이어도 실행시간 80% 감소는 아님 |
| 안정성 | 평균·상위 지연·최악 입력 | 영상에 따라 실행 시간이 달라질 수 있음 |
| 전력·에너지 | 실측/추정 여부, 보드/칩/가속기 범위, J/inference | 빠르더라도 전력이 늘면 에너지는 나빠질 수 있음 |
| 처리량 표기 | 실제 실행 ops와 dense-equivalent ops 구분 | 희소 가속기의 높은 GOPS가 물리 연산량과 다를 수 있음 |

**이 조사에서의 판단:** 단순 CNN을 사용해도 유효한 하드웨어 연구는 가능하다. 다만 “CNN을 FPGA에 올렸다”, “INT8을 썼다”, “0을 건너뛴다”만으로는 최근 연구와 구별하기 어렵다. 우리 장치의 제약 아래에서 어떤 병목을 어떻게 줄였고, 정확도·시간·자원·에너지 사이에 어떤 교환이 있었는지를 보여주는 것이 핵심이다.

## 추가 확인에 사용할 공식 자료

- [CVF Open Access](https://openaccess.thecvf.com/)
- [FCCM 2026 공식 프로그램](https://www.fccm.org/past/2026/fccm-26-program/)
- [FCCM 2025 공식 프로그램](https://www.fccm.org/fccm-2025-program/)
- [FPL 2025 공식 프로그램](https://2025.fpl.org/program/program/)
- [ASP-DAC 2026 공식 초록](https://www.aspdac.com/aspdac2026/program/program-abstract.html)
- [FlexViT 저자 기관 기록 — FPL 2026](https://eprints.gla.ac.uk/388568/)
- [FlexViT 저자 코드](https://github.com/gicLAB/FlexViT)
- [KANELÉ 저자 코드](https://github.com/Duchstf/KANELE)
- [HGQ2 저자 코드](https://github.com/calad0i/HGQ2)

