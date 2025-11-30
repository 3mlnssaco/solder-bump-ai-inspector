# 물리 기반 X-ray 시뮬레이션을 활용한 웨이퍼 레벨 패키징 솔더 범프 결함 검출 시스템

## Physics-Based X-ray Simulation for Wafer-Level Packaging Solder Bump Defect Detection

**연구 기간**: 2025년 12월
**프로젝트 위치**: `/Users/3mln_xx/dev_code/semi_final/solder_bump_desktop/`

---

# 목차

1. [연구 개요](#1-연구-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [물리 기반 X-ray 시뮬레이터](#3-물리-기반-x-ray-시뮬레이터)
4. [데이터셋](#4-데이터셋)
5. [모델 학습](#5-모델-학습)
6. [실험 결과](#6-실험-결과)
7. [시각화 자료](#7-시각화-자료)
8. [결론](#8-결론)
9. [파일 구조](#9-파일-구조)

---

# 1. 연구 개요

## 1.1 연구 배경
- 웨이퍼 레벨 패키징(WLP) 기술의 중요성 증가
- 솔더 범프 결함이 제품 신뢰성에 미치는 영향
- 기존 X-ray 검사의 한계 (수동 분석, 높은 비용)

## 1.2 연구 목적
**물리 기반 시뮬레이션으로 합성 데이터를 생성하고, YOLOv8 딥러닝 모델로 실시간 결함 검출 시스템 개발**

## 1.3 주요 성과

| 지표 | 결과 |
|------|------|
| **mAP@50** | **99.5%** |
| **mAP@50-95** | **99.5%** |
| **Precision** | **99.93%** |
| **Recall** | **99.95%** |
| **F1-Score** | **99.94%** |

---

# 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    전체 시스템 파이프라인                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  물리 기반    │    │   YOLOv8    │    │   결함 검출   │      │
│  │ X-ray 시뮬레이터│ → │  모델 학습   │ → │   & 분류     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                  │                   │               │
│         ▼                  ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Beer-Lambert │    │  98,000개    │    │ 6개 클래스   │      │
│  │   법칙 적용   │    │ 바운딩 박스  │    │  실시간 검출  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 3. 물리 기반 X-ray 시뮬레이터

## 3.1 Beer-Lambert 법칙

X-ray가 물질을 통과할 때의 감쇠를 모델링:

$$I = I_0 \times e^{-\mu \rho t}$$

| 변수 | 설명 | 단위 |
|------|------|------|
| I | 투과 후 강도 | - |
| I₀ | 초기 강도 | - |
| μ | 질량 감쇠 계수 | cm²/g |
| ρ | 밀도 | g/cm³ |
| t | 두께 | cm |

## 3.2 재료별 물성치

| 재료 | 밀도 (g/cm³) | μ @50keV (cm²/g) |
|------|-------------|------------------|
| **Solder (Sn-Ag-Cu)** | 7.4 | 5.2 |
| **Silicon** | 2.33 | 0.85 |
| **Copper** | 8.96 | 3.8 |
| **Void (Air)** | 0.001 | 0.02 |

## 3.3 노이즈 모델

- **Poisson 노이즈**: 양자 통계적 특성
- **Gaussian 노이즈**: 전자 회로 노이즈
- **결합**: σ² = σ_poisson² + σ_gaussian²

## 3.4 시뮬레이터 코드 위치
```
train/physics_xray_simulator.py
```

---

# 4. 데이터셋

## 4.1 웨이퍼 시뮬레이션 파라미터

| 파라미터 | 값 |
|---------|-----|
| 웨이퍼 크기 | 6인치 (150mm) |
| 범프 피치 | 150μm |
| 다이 크기 | 2.1mm × 2.1mm |
| **다이당 범프 수** | **14 × 14 = 196개** |
| 범프 직경 | 80μm |
| 범프 높이 | 60μm |

## 4.2 데이터셋 구성

| 구분 | 이미지 수 | 바운딩 박스 수 |
|------|----------|---------------|
| **Train** | 350 | 68,600 |
| **Validation** | 100 | 19,600 |
| **Test** | 50 | 9,800 |
| **Total** | **500** | **98,000** |

## 4.3 클래스별 분포

| 클래스 | Train | Validation | 설명 |
|--------|-------|------------|------|
| **Normal** | 49,350 | 14,100 | 정상 솔더 범프 |
| **Void** | 8,050 | 2,300 | 내부 공동 |
| **Bridge** | 3,850 | 1,100 | 인접 범프 연결 |
| **HiP** | 2,450 | 700 | Head-in-Pillow |
| **ColdJoint** | 2,450 | 700 | 냉땜 |
| **Crack** | 2,450 | 700 | 균열 |

## 4.4 데이터셋 위치
```
data/xray/
├── train/images/     # 350장
├── valid/images/     # 100장
├── test/images/      # 50장
└── data.yaml         # 설정 파일
```

## 4.5 샘플 이미지

### 시뮬레이션된 다이 이미지 (14×14 범프 어레이)
![Sample Die](data/xray/train/images/train_000000.png)

**이미지 특징:**
- 정상 범프: 균일한 원형, 어두운 색상
- Void: 범프 내부에 밝은 영역
- Bridge: 인접 범프 간 연결된 부분
- Crack: 범프에 선형 패턴

---

# 5. 모델 학습

## 5.1 YOLOv8 아키텍처

```
Model: YOLOv8n (nano)
├── Parameters: 3,012,018
├── GFLOPs: 8.2
├── Layers: 129
└── Input Size: 640 × 640
```

### 주요 모듈
- **Backbone**: CSPDarknet with C2f blocks
- **Neck**: FPN + PAN
- **Head**: Decoupled detection head
- **Loss**: CIoU + BCE + DFL

## 5.2 학습 설정

| 하이퍼파라미터 | 값 |
|--------------|-----|
| **Epochs** | 15 (→ 30 확장 학습 중) |
| **Batch Size** | 4 |
| **Image Size** | 640 × 640 |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 0.0005 |
| **Momentum** | 0.9 |
| **Device** | CPU (Apple M3 Max) |
| **Patience** | 15 (Early Stopping) |

## 5.3 데이터 증강

| 증강 기법 | 설정값 |
|----------|--------|
| Mosaic | 1.0 |
| Horizontal Flip | 0.5 |
| HSV-Hue | 0.015 |
| HSV-Saturation | 0.7 |
| HSV-Value | 0.4 |
| Scale | 0.5 |
| Translation | 0.1 |
| Random Erasing | 0.4 |

## 5.4 학습 결과 위치
```
runs/xray_detection/physics_resume/
├── weights/
│   ├── best.pt      # 최고 성능 모델
│   └── last.pt      # 최종 모델
├── results.csv      # 학습 로그
└── results.png      # 학습 곡선
```

---

# 6. 실험 결과

## 6.1 전체 성능

| 지표 | 값 | 비고 |
|------|------|------|
| **Precision** | 99.93% | 정밀도 |
| **Recall** | 99.95% | 재현율 |
| **mAP@50** | 99.50% | IoU=0.5 기준 |
| **mAP@50-95** | 99.50% | IoU=0.5~0.95 평균 |
| **F1-Score** | 99.94% | 조화 평균 |

## 6.2 클래스별 성능

| 클래스 | Precision | Recall | AP@50 |
|--------|-----------|--------|-------|
| **Void** | 100.0% | 100.0% | 99.5% |
| **Bridge** | 100.0% | 100.0% | 99.5% |
| **HiP** | 99.7% | 100.0% | 99.5% |
| **ColdJoint** | 99.9% | 100.0% | 99.5% |
| **Crack** | 100.0% | 99.7% | 99.5% |
| **Normal** | 100.0% | 100.0% | 99.5% |

## 6.3 학습 곡선 (Epoch별 mAP@50)

```
Epoch  1: ████░░░░░░░░░░░░░░░░  37.4%
Epoch  5: ███████████████░░░░░  74.2%
Epoch 10: ███████████████████░  97.5%
Epoch 15: ████████████████████  99.5%
```

| Epoch | mAP@50 | Box Loss | Cls Loss |
|-------|--------|----------|----------|
| 1 | 0.374 | 0.326 | 1.187 |
| 5 | 0.742 | 0.338 | 0.648 |
| 10 | 0.975 | 0.224 | 0.474 |
| 15 | 0.995 | 0.206 | 0.350 |

## 6.4 추론 속도

| 단계 | 시간 (ms) |
|------|----------|
| Preprocess | 0.8 |
| Inference | 65.3 |
| Postprocess | 19.5 |
| **Total** | **85.6** |

- **처리 속도**: ~11.7 FPS (CPU)
- **GPU 예상**: 60+ FPS

---

# 7. 시각화 자료

## 7.1 학습 곡선 (Training Curves)
**파일**: `runs/xray_detection/physics_resume/results.png`

- 모든 Loss가 안정적으로 수렴
- mAP@50이 Epoch 10 이후 99% 이상 유지
- Precision/Recall 모두 99% 이상

## 7.2 혼동 행렬 (Confusion Matrix)
**파일**: `runs/xray_detection/physics_resume/confusion_matrix_normalized.png`

- 모든 클래스에서 **100% 정확도**
- 클래스 간 혼동 없음
- 대각선에만 값이 집중

## 7.3 PR 곡선 (Precision-Recall Curve)
**파일**: `runs/xray_detection/physics_resume/BoxPR_curve.png`

- 모든 클래스: **AP = 0.995**
- PR 곡선이 (1.0, 1.0)에 근접
- 완벽에 가까운 검출 성능

## 7.4 레이블 분포
**파일**: `runs/xray_detection/physics_resume/labels.jpg`

- Normal: 49,350개 (가장 많음)
- Void: 8,050개
- Bridge: 3,850개
- HiP/ColdJoint/Crack: 각 2,450개

## 7.5 검출 결과 예시
**파일**: `runs/xray_detection/physics_resume/val_batch0_pred.jpg`

- 실제 검출 결과 시각화
- 각 범프에 클래스 레이블 표시
- 신뢰도 점수 포함

---

# 8. 결론

## 8.1 연구 성과

1. **물리 기반 시뮬레이터 개발**
   - Beer-Lambert 법칙 기반 현실적 X-ray 이미지 생성
   - 98,000개 바운딩 박스 자동 생성

2. **높은 검출 성능 달성**
   - mAP@50: **99.5%**
   - 모든 결함 클래스에서 **99.5% 이상** AP

3. **실시간 처리 가능**
   - CPU에서 ~12 FPS
   - GPU 사용 시 실시간 검사 가능

## 8.2 기술적 기여

| 기여 항목 | 내용 |
|----------|------|
| 데이터 생성 | 실제 X-ray 장비 없이 대규모 학습 데이터 생성 |
| 라벨링 정확도 | 100% 정확한 자동 어노테이션 |
| 결함 다양성 | 6종 결함 클래스 시뮬레이션 |
| 물리적 정확성 | Beer-Lambert 법칙 기반 현실적 감쇠 모델 |

## 8.3 한계점 및 향후 연구

| 한계점 | 해결 방향 |
|--------|----------|
| Sim-to-Real Gap | Domain Adaptation 적용 |
| 결함 다양성 | 실제 데이터로 Fine-tuning |
| 재료 복잡성 | 다층 구조 시뮬레이션 추가 |

---

# 9. 파일 구조

```
solder_bump_desktop/
│
├── 📁 data/xray/                          # 데이터셋
│   ├── train/images/                      # 학습 이미지 (350장)
│   ├── valid/images/                      # 검증 이미지 (100장)
│   ├── test/images/                       # 테스트 이미지 (50장)
│   └── data.yaml                          # 데이터셋 설정
│
├── 📁 runs/xray_detection/                # 학습 결과
│   └── physics_resume/
│       ├── weights/
│       │   ├── best.pt                    # ⭐ 최고 성능 모델
│       │   └── last.pt
│       ├── results.csv                    # 학습 로그
│       ├── results.png                    # 📊 학습 곡선
│       ├── confusion_matrix_normalized.png # 📊 혼동 행렬
│       ├── BoxPR_curve.png                # 📊 PR 곡선
│       ├── BoxF1_curve.png                # 📊 F1 곡선
│       ├── labels.jpg                     # 📊 레이블 분포
│       └── val_batch*_pred.jpg            # 📊 검출 결과
│
├── 📁 train/
│   └── physics_xray_simulator.py          # 물리 시뮬레이터
│
├── evaluate_model.py                       # 평가 스크립트
├── RESEARCH_REPORT.md                      # 연구 레포트
└── PRESENTATION_REPORT.md                  # 📋 발표용 레포트 (현재 파일)
```

---

# 발표용 주요 그림 링크

| 용도 | 파일 경로 |
|------|----------|
| **학습 곡선** | `runs/xray_detection/physics_resume/results.png` |
| **혼동 행렬** | `runs/xray_detection/physics_resume/confusion_matrix_normalized.png` |
| **PR 곡선** | `runs/xray_detection/physics_resume/BoxPR_curve.png` |
| **F1 곡선** | `runs/xray_detection/physics_resume/BoxF1_curve.png` |
| **레이블 분포** | `runs/xray_detection/physics_resume/labels.jpg` |
| **검출 결과 1** | `runs/xray_detection/physics_resume/val_batch0_pred.jpg` |
| **검출 결과 2** | `runs/xray_detection/physics_resume/val_batch1_pred.jpg` |
| **샘플 이미지** | `data/xray/train/images/train_000000.png` |

---

# 핵심 수치 요약 (발표용)

```
┌────────────────────────────────────────────────────┐
│              🎯 최종 성능 지표                      │
├────────────────────────────────────────────────────┤
│                                                    │
│     Precision    ████████████████████  99.93%     │
│     Recall       ████████████████████  99.95%     │
│     mAP@50       ████████████████████  99.50%     │
│     F1-Score     ████████████████████  99.94%     │
│                                                    │
├────────────────────────────────────────────────────┤
│              📊 데이터셋 규모                       │
├────────────────────────────────────────────────────┤
│     이미지 수           500장                      │
│     바운딩 박스      98,000개                      │
│     결함 클래스          6종                       │
│                                                    │
├────────────────────────────────────────────────────┤
│              ⚡ 처리 속도                          │
├────────────────────────────────────────────────────┤
│     CPU (M3 Max)    ~12 FPS                       │
│     GPU (예상)      ~60 FPS                       │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

**작성일**: 2025년 12월 1일
**모델**: YOLOv8n (physics_resume)
**프로젝트**: Solder Bump X-ray Inspection System
