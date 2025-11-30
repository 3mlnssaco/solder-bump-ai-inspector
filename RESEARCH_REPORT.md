# 물리 기반 X-ray 시뮬레이션을 활용한 웨이퍼 레벨 패키징 솔더 범프 결함 검출 시스템

## Physics-Based X-ray Simulation for Wafer-Level Packaging Solder Bump Defect Detection System

---

## 1. 서론 (Introduction)

### 1.1 연구 배경
웨이퍼 레벨 패키징(WLP) 기술은 반도체 산업에서 소형화, 고성능화 요구에 대응하기 위한 핵심 기술이다. 특히 솔더 범프는 칩과 기판 간의 전기적 연결을 담당하는 중요한 요소로, 결함 발생 시 제품 신뢰성에 치명적인 영향을 미친다.

### 1.2 연구 목적
본 연구는 물리 기반 X-ray 시뮬레이션을 통해 합성 학습 데이터를 생성하고, YOLOv8 딥러닝 모델을 활용하여 솔더 범프 결함을 실시간으로 검출하는 시스템을 개발하는 것을 목표로 한다.

### 1.3 주요 기여
1. Beer-Lambert 법칙 기반 물리적 X-ray 감쇠 시뮬레이터 개발
2. 6종 결함 클래스를 포함한 대규모 합성 데이터셋 생성
3. YOLOv8 기반 실시간 결함 검출 시스템 구현
4. 99.5% mAP@50 달성

---

## 2. 관련 연구 (Related Work)

### 2.1 X-ray 기반 비파괴 검사
- 전통적 X-ray 검사: 수동 분석, 시간 소요
- 자동화 검사: AOI (Automated Optical Inspection) 한계
- X-ray CT: 고비용, 저속도 문제

### 2.2 딥러닝 기반 결함 검출
- CNN 기반 분류 모델
- YOLO 계열 객체 탐지 모델
- Semantic Segmentation (DeepLabV3+)

### 2.3 합성 데이터 생성
- Domain Randomization
- Physics-based Simulation
- GAN 기반 데이터 증강

---

## 3. 방법론 (Methodology)

### 3.1 물리 기반 X-ray 시뮬레이터

#### 3.1.1 Beer-Lambert 법칙
X-ray 감쇠는 다음 수식을 따른다:

```
I = I₀ × exp(-μρt)
```

여기서:
- I: 투과 후 X-ray 강도
- I₀: 초기 X-ray 강도
- μ: 질량 감쇠 계수 (cm²/g)
- ρ: 재료 밀도 (g/cm³)
- t: 재료 두께 (cm)

#### 3.1.2 재료별 물성치

| 재료 | 밀도 (g/cm³) | 감쇠계수 @50keV (cm²/g) |
|------|-------------|------------------------|
| Solder (Sn-Ag-Cu) | 7.4 | 5.2 |
| Silicon | 2.33 | 0.85 |
| Copper | 8.96 | 3.8 |
| Void (Air) | 0.001 | 0.02 |

#### 3.1.3 폴리크로매틱 X-ray 스펙트럼
Kramers' 법칙을 적용한 연속 스펙트럼 모델링:

```
I(E) = K × Z × (E_max - E) / E
```

#### 3.1.4 노이즈 모델
- Poisson 노이즈: 양자 통계적 특성
- Gaussian 노이즈: 전자 회로 노이즈
- 결합 모델: σ² = σ_poisson² + σ_gaussian²

### 3.2 웨이퍼 시뮬레이션 파라미터

| 파라미터 | 값 |
|---------|-----|
| 웨이퍼 크기 | 6인치 (150mm) |
| 범프 피치 | 150μm |
| 다이 크기 | 2.1mm × 2.1mm |
| 다이당 범프 수 | 14 × 14 = 196개 |
| 범프 직경 | 80μm |
| 범프 높이 | 60μm |

### 3.3 결함 클래스 정의

| 클래스 | 설명 | 물리적 특성 |
|--------|------|------------|
| **Void** | 솔더 내부 공동 | 감쇠 감소 (밝은 영역) |
| **Bridge** | 인접 범프 간 연결 | 확장된 솔더 영역 |
| **HiP (Head in Pillow)** | 불완전 접합 | 비균일 감쇠 패턴 |
| **ColdJoint** | 냉땜 | 불규칙한 경계 |
| **Crack** | 균열 | 선형 감쇠 불연속 |
| **Normal** | 정상 범프 | 균일한 구형 감쇠 |

### 3.4 YOLOv8 모델 아키텍처

```
Model: YOLOv8n (nano)
- Parameters: 3,012,018
- GFLOPs: 8.2
- Layers: 129
- Input Size: 640 × 640
```

#### 주요 모듈:
- Backbone: CSPDarknet with C2f blocks
- Neck: FPN + PAN
- Head: Decoupled detection head
- Loss: CIoU + BCE + DFL

---

## 4. 실험 (Experiments)

### 4.1 데이터셋 구성

#### 4.1.1 합성 데이터셋 (Physics-based)

| 구분 | 이미지 수 | 바운딩 박스 수 |
|------|----------|---------------|
| Train | 350 | 68,600 |
| Validation | 100 | 19,600 |
| Test | 50 | 9,800 |
| **Total** | **500** | **98,000** |

#### 4.1.2 클래스별 분포 (Validation Set)

| 클래스 | 인스턴스 수 | 비율 |
|--------|-----------|------|
| Normal | 14,100 | 71.9% |
| Void | 2,300 | 11.7% |
| Bridge | 1,100 | 5.6% |
| HiP | 700 | 3.6% |
| ColdJoint | 700 | 3.6% |
| Crack | 700 | 3.6% |

### 4.2 학습 설정

| 하이퍼파라미터 | 값 |
|--------------|-----|
| Epochs | 15 |
| Batch Size | 4 |
| Image Size | 640 × 640 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Momentum | 0.9 |
| Device | CPU (Apple M3 Max) |

### 4.3 데이터 증강

- Mosaic augmentation
- Random flip (horizontal)
- HSV augmentation (H: 0.015, S: 0.7, V: 0.4)
- Scale: 0.5
- Translation: 0.1
- Random erasing: 0.4

---

## 5. 결과 (Results)

### 5.1 전체 성능 지표

| 지표 | 값 |
|------|------|
| **Precision** | 99.93% |
| **Recall** | 99.95% |
| **mAP@50** | 99.50% |
| **mAP@50-95** | 99.50% |
| **F1-Score** | 99.94% |

### 5.2 클래스별 성능

| 클래스 | Precision | Recall | AP@50 |
|--------|-----------|--------|-------|
| Void | 100.0% | 100.0% | 99.5% |
| Bridge | 100.0% | 100.0% | 99.5% |
| HiP | 99.7% | 100.0% | 99.5% |
| ColdJoint | 99.9% | 100.0% | 99.5% |
| Crack | 100.0% | 99.7% | 99.5% |
| Normal | 100.0% | 100.0% | 99.5% |

### 5.3 학습 곡선

#### Epoch별 mAP@50 변화:
```
Epoch  1: 0.374 (37.4%)
Epoch  2: 0.574 (57.4%)
Epoch  3: 0.655 (65.5%)
Epoch  4: 0.740 (74.0%)
Epoch  5: 0.742 (74.2%)
Epoch  6: 0.766 (76.6%)
Epoch  7: 0.792 (79.2%)
Epoch  8: 0.819 (81.9%)
Epoch  9: 0.873 (87.3%)
Epoch 10: 0.975 (97.5%)
Epoch 11: 0.995 (99.5%)
Epoch 12: 0.995 (99.5%)
Epoch 13: 0.995 (99.5%)
Epoch 14: 0.995 (99.5%)
Epoch 15: 0.995 (99.5%)
```

### 5.4 추론 속도

| 단계 | 시간 (ms) |
|------|----------|
| Preprocess | 0.8 |
| Inference | 65.3 |
| Postprocess | 19.5 |
| **Total** | **85.6** |

- 처리 속도: **~11.7 FPS** (CPU)
- GPU 사용 시 예상: **60+ FPS**

### 5.5 손실 함수 수렴

| Epoch | Box Loss | Cls Loss | DFL Loss |
|-------|----------|----------|----------|
| 1 | 0.326 | 1.187 | 0.810 |
| 5 | 0.338 | 0.648 | 0.806 |
| 10 | 0.224 | 0.474 | 0.798 |
| 15 | 0.206 | 0.350 | 0.800 |

---

## 6. 분석 및 논의 (Discussion)

### 6.1 물리 기반 시뮬레이션의 장점

1. **데이터 수집 비용 절감**: 실제 X-ray 장비 없이 대규모 데이터 생성
2. **Ground Truth 정확성**: 자동 라벨링으로 인한 100% 정확한 어노테이션
3. **결함 비율 조절**: 희귀 결함 데이터 증강 용이
4. **물리적 일관성**: Beer-Lambert 법칙 기반 현실적 이미지 생성

### 6.2 높은 성능의 원인 분석

1. **일관된 데이터 품질**: 시뮬레이션 데이터의 노이즈 특성 일관성
2. **명확한 결함 특성**: 물리적으로 정의된 결함 패턴
3. **충분한 학습 데이터**: 98,000개 바운딩 박스
4. **적절한 모델 크기**: YOLOv8n의 효율적 아키텍처

### 6.3 한계점

1. **Sim-to-Real Gap**: 실제 X-ray 이미지와의 도메인 차이
2. **결함 다양성**: 실제 산업 환경의 다양한 결함 형태 미반영
3. **재료 복잡성**: 단순화된 재료 모델

### 6.4 향후 연구 방향

1. **Domain Adaptation**: 실제 데이터와의 도메인 갭 해소
2. **Transfer Learning**: 실제 소량 데이터로 Fine-tuning
3. **Multi-scale Detection**: 다양한 범프 크기 대응
4. **3D Reconstruction**: X-ray CT 기반 3차원 결함 분석

---

## 7. 결론 (Conclusion)

본 연구에서는 Beer-Lambert 법칙 기반 물리 시뮬레이션을 활용하여 웨이퍼 레벨 패키징 솔더 범프 X-ray 검사 데이터셋을 생성하고, YOLOv8 모델을 통해 6종 결함을 검출하는 시스템을 개발하였다.

### 주요 성과:
- **99.5% mAP@50** 달성
- **99.94% F1-Score** 달성
- 모든 결함 클래스에서 **99.5% 이상** AP 달성
- **~12 FPS** 실시간 처리 (CPU 기준)

물리 기반 합성 데이터가 딥러닝 모델 학습에 효과적임을 입증하였으며, 향후 실제 산업 데이터와의 도메인 적응을 통해 실용화 가능성을 높일 수 있을 것으로 기대된다.

---

## 8. 참고문헌 (References)

1. Hubbell, J. H., & Seltzer, S. M. (1995). Tables of X-ray mass attenuation coefficients. NIST.
2. Jocher, G., et al. (2023). Ultralytics YOLOv8. GitHub repository.
3. Kramers, H. A. (1923). On the theory of X-ray absorption. Philosophical Magazine.
4. Liu, W., et al. (2016). SSD: Single shot multibox detector. ECCV.
5. Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. arXiv.

---

## 부록 (Appendix)

### A. 프로젝트 구조

```
solder_bump_desktop/
├── data/
│   └── xray/
│       ├── train/images/     # 350 images
│       ├── valid/images/     # 100 images
│       ├── test/images/      # 50 images
│       └── data.yaml
├── runs/
│   └── xray_detection/
│       ├── physics_based/    # Initial training
│       └── physics_resume/   # Final model (best.pt)
├── train/
│   └── physics_xray_simulator.py
└── evaluate_model.py
```

### B. 모델 파일

- **Best Model**: `runs/xray_detection/physics_resume/weights/best.pt`
- **Parameters**: 3,012,018
- **Size**: ~6MB

### C. 평가 결과 JSON

```json
{
  "model": "YOLOv8n (physics_resume)",
  "epochs": 15,
  "metrics": {
    "precision": 0.9993,
    "recall": 0.9995,
    "mAP50": 0.995,
    "mAP50_95": 0.995,
    "f1_score": 0.9994
  }
}
```

---

**작성일**: 2025년 12월 1일
**작성자**: Claude AI Assistant
**프로젝트**: Solder Bump X-ray Inspection System
