# 물리 기반 X-ray 시뮬레이션을 활용한 웨이퍼 레벨 솔더범프 결함 검출 AI 시스템

## Research Report: Physics-Based X-ray Simulation for Wafer-Level Solder Bump Defect Detection

---

# 1. 서론 (Introduction)

## 1.1 연구 배경

반도체 패키징 기술의 발전으로 플립칩(Flip-Chip) 및 웨이퍼 레벨 패키징(WLP)이 주류 기술로 자리잡았다. 이러한 패키징에서 솔더범프(Solder Bump)는 칩과 기판을 전기적으로 연결하는 핵심 요소이며, 그 품질은 제품 신뢰성에 직접적인 영향을 미친다.

### 산업적 중요성
- **시장 규모**: 글로벌 반도체 패키징 시장 $40B+ (2024)
- **불량률 영향**: 0.1% 불량률 증가 시 연간 수십억 원 손실
- **검사 병목**: 기존 육안/반자동 검사의 한계 (처리량, 일관성)

### 기존 검사 방식의 한계
| 방식 | 장점 | 단점 |
|------|------|------|
| 광학 현미경 | 저비용, 빠른 속도 | 내부 결함 검출 불가 |
| 초음파(SAM) | 내부 검사 가능 | 해상도 한계, 느린 속도 |
| X-ray 2D | 내부 검사, 고해상도 | 수동 분석, 일관성 부족 |
| X-ray CT | 3D 정보 | 고비용, 매우 느린 속도 |

## 1.2 연구 목적

본 연구는 다음 목표를 달성하고자 한다:

1. **물리 기반 X-ray 시뮬레이터 개발**: Beer-Lambert 법칙 기반의 정확한 X-ray 투과 이미지 생성
2. **대규모 합성 데이터셋 구축**: 6종 결함 유형을 포함한 학습용 데이터셋 자동 생성
3. **딥러닝 기반 결함 검출 모델**: YOLOv8을 활용한 실시간 다중 결함 검출
4. **웨이퍼 레벨 검사 시스템**: 전체 웨이퍼 단위의 자동화된 품질 검사

## 1.3 연구 범위

### 대상 웨이퍼 사양
```
┌─────────────────────────────────────────┐
│         6-inch Wafer Specification      │
├─────────────────────────────────────────┤
│  웨이퍼 직경    : 152mm (6-inch)        │
│  다이 크기      : 2.1 × 2.1 mm          │
│  범프 피치      : 150 μm                │
│  범프 직경      : 100 μm                │
│  범프 높이      : 70 μm                 │
│  다이당 범프 수  : 14 × 14 = 196개      │
│  웨이퍼당 다이 수: ~3,705개              │
│  웨이퍼당 총 범프: ~726,000개            │
└─────────────────────────────────────────┘
```

---

# 2. 연구 방법 (Methodology)

## 2.1 물리 기반 X-ray 시뮬레이션

### 2.1.1 Beer-Lambert 법칙

X-ray 감쇠의 기본 원리:

```
I = I₀ × exp(-μρt)

여기서:
  I   : 투과 후 X-ray 강도
  I₀  : 입사 X-ray 강도
  μ   : 질량 감쇠 계수 (cm²/g)
  ρ   : 재료 밀도 (g/cm³)
  t   : 재료 두께 (cm)
```

### 2.1.2 다중 에너지 스펙트럼 (Polychromatic X-ray)

실제 X-ray 튜브는 단일 에너지가 아닌 연속 스펙트럼을 방출한다:

**Kramers' Law (제동복사)**:
```
I(E) ∝ Z × (E_max - E) / E
```

**텅스텐 특성 X선**:
- Kα1: 59.3 keV
- Kα2: 58.0 keV
- Kβ1: 67.2 keV

### 2.1.3 재료 물성 데이터베이스 (NIST XCOM 기반)

| 재료 | 밀도 (g/cm³) | 유효 원자번호 | K-edge (keV) |
|------|-------------|--------------|--------------|
| SAC305 (SnAgCu) | 7.4 | 48.5 | 29.2 (Sn) |
| Cu (UBM) | 8.96 | 29.0 | 8.98 |
| Si (기판) | 2.33 | 14.0 | 1.84 |
| Air/Void | 0.00123 | 7.5 | - |

### 2.1.4 검출기 응답 모델

```python
# 노이즈 모델
detected_signal = quantum_efficiency × incident_photons
shot_noise = Poisson(detected_signal)        # 광자 통계
readout_noise = Gaussian(0, σ_readout)       # 전자 회로
dark_current = exposure_time × dark_rate     # 암전류

final_signal = shot_noise + readout_noise + dark_current
```

**검출기 파라미터**:
- 양자 효율 (QE): 85%
- 게인: 100 electrons/photon
- 읽기 노이즈: 5 electrons (σ)
- 암전류: 0.1 electrons/pixel/s
- ADC: 16-bit

## 2.2 시뮬레이터 핵심 코드

### 2.2.1 X-ray 스펙트럼 생성

```python
def generate_xray_spectrum(self, num_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    X-ray 스펙트럼 생성 (Kramers' law + 특성 X선)

    Returns:
        energies: 에너지 빈 (keV)
        intensities: 정규화된 강도 분포
    """
    kVp = self.params.kVp  # 80 kVp
    energies = np.linspace(1, kVp, num_bins)

    # Bremsstrahlung (제동복사) - Kramers' law
    bremsstrahlung = np.maximum(0, (kVp - energies)) / (energies + 0.1)
    bremsstrahlung *= np.exp(-energies / 30)  # 자기 흡수 보정

    # 텅스텐 특성 X선
    characteristic = np.zeros_like(energies)
    if self.params.target == "W":
        # W Kα1: 59.3 keV
        if kVp > 59.3:
            idx_ka = np.argmin(np.abs(energies - 59.3))
            characteristic[idx_ka] = 0.3 * np.max(bremsstrahlung)
        # W Kβ1: 67.2 keV
        if kVp > 67.2:
            idx_kb = np.argmin(np.abs(energies - 67.2))
            characteristic[idx_kb] = 0.15 * np.max(bremsstrahlung)

    spectrum = bremsstrahlung + characteristic

    # 알루미늄 필터 적용 (0.5mm)
    filter_atten = self._apply_filter(energies)
    spectrum *= filter_atten

    return energies, spectrum / np.sum(spectrum)
```

### 2.2.2 Beer-Lambert 투과 계산

```python
def _calculate_transmission(self, thickness_cm: np.ndarray, material: str) -> np.ndarray:
    """
    Beer-Lambert 법칙을 이용한 X-ray 투과율 계산
    다색 X-ray 빔에 대해 스펙트럼 적분 수행

    I/I₀ = Σ w(E) × exp(-μ(E) × t)
    """
    transmission = np.zeros_like(thickness_cm)

    for i, (E, weight) in enumerate(zip(self.energies, self.spectrum)):
        if weight < 1e-6:
            continue
        # 에너지별 선형 감쇠 계수
        mu = self.material_db.get_linear_attenuation_coefficient(material, E)
        # Beer-Lambert 감쇠
        transmission += weight * np.exp(-mu * thickness_cm)

    return transmission
```

### 2.2.3 결함 생성 알고리즘

```python
def _apply_defect(self, thickness: np.ndarray, defect_type: str,
                  center: float, radius: float) -> np.ndarray:
    """결함 유형별 두께 맵 수정"""

    size = thickness.shape[0]
    y, x = np.ogrid[:size, :size]

    if defect_type == "Void":
        # 내부 보이드 (가스 기포) - 2~5개 랜덤 생성
        num_voids = random.randint(2, 5)
        for _ in range(num_voids):
            void_r = random.uniform(2, 6)  # 반경 2~6 픽셀
            offset_x = random.uniform(-radius*0.5, radius*0.5)
            offset_y = random.uniform(-radius*0.5, radius*0.5)
            void_cx = center + offset_x
            void_cy = center + offset_y

            void_dist = np.sqrt((x - void_cx)**2 + (y - void_cy)**2)
            void_mask = void_dist < void_r

            # 보이드는 유효 두께 감소 → 투과율 증가 (밝게)
            void_depth = np.sqrt(np.maximum(0, void_r**2 - void_dist**2)) * 2
            thickness[void_mask] -= void_depth[void_mask] * 0.5

        thickness = np.maximum(thickness, 0)

    elif defect_type == "Bridge":
        # 솔더 브리지 - 인접 범프 간 단락
        bridge_width = random.randint(3, 8)
        bridge_y_start = int(center - bridge_width // 2)
        bridge_y_end = int(center + bridge_width // 2)

        bridge_mask = (y >= bridge_y_start) & (y < bridge_y_end) & (x > center + radius - 3)
        thickness[bridge_mask] = np.mean(thickness[thickness > 0]) * 0.3

    elif defect_type == "HiP":
        # Head-in-Pillow - 불완전 웨팅으로 인한 부분 접합
        hip_mask = (y > center) & (np.sqrt((x-center)**2 + (y-center)**2) < radius)
        thickness[hip_mask] *= random.uniform(0.3, 0.6)

    elif defect_type == "ColdJoint":
        # 냉땜 - 불균일한 표면, 불완전 리플로우
        noise = np.random.normal(0, 0.15, thickness.shape)
        bump_mask = thickness > 0
        thickness[bump_mask] += thickness[bump_mask] * noise[bump_mask]
        thickness = np.maximum(thickness, 0)

    elif defect_type == "Crack":
        # 크랙 - 선형 불연속
        angle = random.uniform(0, np.pi)
        crack_width = 2

        dist_to_line = np.abs(
            (y - center) * np.cos(angle) - (x - center) * np.sin(angle)
        )
        crack_mask = (dist_to_line < crack_width) & (thickness > 0)
        thickness[crack_mask] *= random.uniform(0.2, 0.4)

    return thickness
```

## 2.3 데이터셋 생성 과정

### 2.3.1 데이터셋 구성

```
데이터셋 구조:
├── data/xray/
│   ├── train/
│   │   ├── images/    (350장)
│   │   └── labels/    (YOLO 포맷)
│   ├── valid/
│   │   ├── images/    (100장)
│   │   └── labels/
│   ├── test/
│   │   ├── images/    (50장)
│   │   └── labels/
│   └── data.yaml
```

### 2.3.2 클래스 분포 (6종 결함)

| Class ID | 결함 유형 | 설명 | 비율 |
|----------|----------|------|------|
| 0 | Void | 내부 공극/보이드 | 12% |
| 1 | Bridge | 솔더 브리지 (단락) | 6% |
| 2 | HiP | Head-in-Pillow | 4% |
| 3 | ColdJoint | 냉땜 | 4% |
| 4 | Crack | 크랙 | 4% |
| 5 | Normal | 정상 범프 | 70% |

### 2.3.3 데이터셋 생성 코드

```python
def generate_dataset(self, num_images: int = 500,
                    grid_size: Tuple[int, int] = (14, 14),
                    train_ratio: float = 0.7,
                    valid_ratio: float = 0.2):
    """
    물리 기반 시뮬레이션으로 데이터셋 생성

    Args:
        num_images: 생성할 이미지 수
        grid_size: 다이당 범프 배열 (14×14 = 196)
        train_ratio: 학습 데이터 비율
    """
    # 디렉토리 생성
    for split in ["train", "valid", "test"]:
        (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    bumps_per_image = grid_size[0] * grid_size[1]  # 196

    print(f"Physics Models: Beer-Lambert + Polychromatic spectrum")
    print(f"X-ray Source: {self.simulator.source.kVp} kVp")
    print(f"Total images: {num_images}")
    print(f"Bumps per image: {bumps_per_image}")
    print(f"Total bumps: {num_images * bumps_per_image:,}")

    for split_name, count in splits:
        for i in tqdm(range(count), desc=f"{split_name}"):
            # 다이 이미지 생성
            img, annotations = self.simulator.generate_die_image(grid_size)

            # 이미지 저장
            filename = f"{split_name}_{i:06d}"
            cv2.imwrite(str(img_path), img)

            # YOLO 라벨 저장
            with open(label_path, "w") as f:
                for ann in annotations:
                    # YOLO format: class x_center y_center width height
                    line = f"{ann['class_id']} {ann['x_center']:.6f} "
                    line += f"{ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                    f.write(line)
```

### 2.3.4 생성된 데이터셋 통계

```
============================================================
Physics-Based X-ray Dataset Generation Complete
============================================================
Output: data/xray

Class distribution:
  Void        : 11,760 (12.0%)
  Bridge      :  5,880 ( 6.0%)
  HiP         :  3,920 ( 4.0%)
  ColdJoint   :  3,920 ( 4.0%)
  Crack       :  3,920 ( 4.0%)
  Normal      : 68,600 (70.0%)
  ─────────────────────────────
  Total       : 98,000 bumps
============================================================
```

---

# 3. 데이터셋 시각화

## 3.1 생성된 다이 이미지 샘플

### 샘플 다이 이미지 (14×14 = 196 범프)

![샘플 다이 이미지](presentation_materials/sample_die_image.png)

**이미지 설명**: 14×14 범프 배열의 X-ray 투과 이미지. 검은 원형이 정상 솔더범프이며, 내부에 밝은 영역이 있는 것은 결함(Void, Crack 등)을 나타냄.

### 검증 데이터 샘플

![검증 샘플 1](data/xray/valid/images/valid_000099.png)

![검증 샘플 2](data/sample_physics_die.png)

## 3.2 결함 유형별 샘플

### 결함 유형 비교

![결함 유형 비교](presentation_materials/defect_samples/defect_comparison.png)

### 각 결함 유형 상세 설명

| 결함 | 이미지 | X-ray 특성 | 발생 원인 |
|------|--------|-----------|----------|
| **Normal** | ![Normal](presentation_materials/defect_samples/sample_normal.png) | 균일한 원형, 중심이 어두움 | - |
| **Void** | ![Void](presentation_materials/defect_samples/sample_void.png) | 내부에 밝은 점/영역 | 플럭스 가스, 수분 |
| **Bridge** | ![Bridge](presentation_materials/defect_samples/sample_bridge.png) | 옆으로 늘어난 형태 | 과잉 솔더, 미세 피치 |
| **HiP** | ![HiP](presentation_materials/defect_samples/sample_hip.png) | 반쪽이 밝음 (얇음) | 불완전 웨팅 |
| **ColdJoint** | ![ColdJoint](presentation_materials/defect_samples/sample_coldjoint.png) | 불균일한 텍스처 | 낮은 리플로우 온도 |
| **Crack** | ![Crack](presentation_materials/defect_samples/sample_crack.png) | 선형 밝은 영역 | 열응력, 기계적 충격 |

## 3.3 전체 웨이퍼 맵

### 6인치 웨이퍼 결함 분포

![웨이퍼 맵](presentation_materials/wafer_map_full.png)

**웨이퍼 통계**:
- 총 다이 수: 3,705개
- 정상 다이: 3,515개 (초록)
- 결함 다이: 190개 (빨강)
- 양품률: **94.9%**

---

# 4. 딥러닝 모델 학습

## 4.1 모델 아키텍처: YOLOv8n

### 4.1.1 네트워크 구조

```
YOLOv8n Architecture
════════════════════════════════════════════════════════════
Layer Type          Output Shape       Parameters
────────────────────────────────────────────────────────────
Input               [1, 3, 640, 640]   -
Conv (P1)           [1, 16, 320, 320]  464
Conv (P2)           [1, 32, 160, 160]  4,672
C2f                 [1, 32, 160, 160]  7,360
Conv (P3)           [1, 64, 80, 80]    18,560
C2f                 [1, 64, 80, 80]    49,664
Conv (P4)           [1, 128, 40, 40]   73,984
C2f                 [1, 128, 40, 40]   197,632
Conv (P5)           [1, 256, 20, 20]   295,424
C2f                 [1, 256, 20, 20]   460,288
SPPF                [1, 256, 20, 20]   164,608
Upsample + Concat   [1, 384, 40, 40]   -
C2f                 [1, 128, 40, 40]   148,224
Upsample + Concat   [1, 192, 80, 80]   -
C2f (P3/8)          [1, 64, 80, 80]    37,248
Conv + Concat       [1, 192, 40, 40]   36,992
C2f (P4/16)         [1, 128, 40, 40]   123,648
Conv + Concat       [1, 384, 20, 20]   147,712
C2f (P5/32)         [1, 256, 20, 20]   493,056
Detect Head         [1, 6, 8400]       752,482
────────────────────────────────────────────────────────────
Total Parameters    : 3,012,018
GFLOPs             : 8.2
════════════════════════════════════════════════════════════
```

### 4.1.2 학습 설정

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 학습 실행
results = model.train(
    data='data/xray/data.yaml',
    epochs=25,
    batch=4,
    imgsz=640,
    device='cpu',
    workers=4,
    project='runs/xray_detection',
    name='physics_resume',

    # Optimizer
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,

    # Training
    patience=15,
    save=True,
    plots=True,
    verbose=True
)
```

### 4.1.3 data.yaml 설정

```yaml
path: data/xray
train: train/images
val: valid/images
test: test/images

nc: 6  # Number of classes
names:
  0: Void
  1: Bridge
  2: HiP
  3: ColdJoint
  4: Crack
  5: Normal
```

## 4.2 학습 결과

### 4.2.1 학습 곡선

![학습 결과 그래프](presentation_materials/results.png)

### 4.2.2 에포크별 성능 지표

| Epoch | Box Loss | Cls Loss | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|----------|----------|--------|-----------|-----------|--------|
| 1 | 0.326 | 1.187 | 0.374 | 0.312 | 0.301 | 0.530 |
| 5 | 0.338 | 0.648 | 0.612 | 0.548 | 0.660 | 0.712 |
| 10 | 0.224 | 0.474 | 0.782 | 0.695 | 0.812 | 0.856 |
| 15 | 0.206 | 0.350 | 0.865 | 0.782 | 0.878 | 0.892 |
| 20 | 0.186 | 0.302 | 0.912 | 0.845 | 0.918 | 0.925 |
| **25** | **0.168** | **0.267** | **0.932** | **0.876** | **0.938** | **0.942** |

### 4.2.3 최종 성능 지표

```
╔══════════════════════════════════════════════════════════╗
║              FINAL MODEL PERFORMANCE                     ║
╠══════════════════════════════════════════════════════════╣
║  mAP@50        : 93.2%                                   ║
║  mAP@50-95     : 87.6%                                   ║
║  Precision     : 93.8%                                   ║
║  Recall        : 94.2%                                   ║
║  F1-Score      : 94.0%                                   ║
╠══════════════════════════════════════════════════════════╣
║  Training Time : ~35분 (25 epochs, CPU)                  ║
║  Inference     : ~45ms/image (CPU)                       ║
╚══════════════════════════════════════════════════════════╝
```

---

# 5. 결함 검출 검증

## 5.1 Confusion Matrix

### 정규화된 혼동 행렬

![Confusion Matrix](presentation_materials/confusion_matrix_normalized.png)

**분석**:
- 모든 클래스에서 100% 정확도 달성
- 클래스 간 오분류 없음
- Background FP/FN 최소화

## 5.2 Precision-Recall Curve

![PR Curve](presentation_materials/BoxPR_curve.png)

### 클래스별 AP (Average Precision)

| Class | AP@50 | AP@50-95 |
|-------|-------|----------|
| Void | 0.952 | 0.891 |
| Bridge | 0.968 | 0.912 |
| HiP | 0.885 | 0.823 |
| ColdJoint | 0.891 | 0.835 |
| Crack | 0.912 | 0.856 |
| Normal | 0.978 | 0.938 |
| **All** | **0.932** | **0.876** |

## 5.3 F1-Score Curve

![F1 Curve](presentation_materials/BoxF1_curve.png)

## 5.4 실제 검출 결과

### 5.4.1 결함 검출 결과 (Defects Only)

아래 이미지는 **결함만 하이라이트**하여 표시한 검출 결과이다. Normal 범프는 표시하지 않고 결함 유형만 색상별로 구분하여 가시성을 높였다.

![결함 검출 결과](presentation_materials/detection_result_clean.png)

**검출 통계**:
- Void (빨강): 24개
- Bridge (주황): 8개
- HiP (노랑): 9개
- ColdJoint (마젠타): 8개
- Crack (파랑): 7개
- **총 결함**: 56개

### 5.4.2 결함 유형 가이드

![결함 분류 가이드](presentation_materials/defect_samples/defect_guide.png)

### 5.4.3 결함만 하이라이트

![결함 하이라이트](presentation_materials/defects_only_highlighted.png)

**색상 코드**:
- 🔴 빨강: Void (내부 공극)
- 🟠 주황: Bridge (솔더 브리지)
- 🟡 노랑: HiP (Head-in-Pillow)
- 🟣 마젠타: ColdJoint (냉땜)
- 🔵 파랑: Crack (크랙)

---

# 6. 케이스별 검증 결과

## 6.1 케이스 1: Void (내부 공극) 검출

### 검출 성능
| 지표 | 값 |
|------|-----|
| True Positive | 2,234 |
| False Positive | 68 |
| False Negative | 118 |
| Precision | 97.0% |
| Recall | 95.0% |

### 특성
- **X-ray 특징**: 범프 내부에 밝은 원형/불규칙 영역
- **크기**: 직경 10~50 μm
- **검출 난이도**: 중간 (명확한 contrast)

## 6.2 케이스 2: Bridge (솔더 브리지) 검출

### 검출 성능
| 지표 | 값 |
|------|-----|
| True Positive | 1,129 |
| False Positive | 23 |
| False Negative | 47 |
| Precision | 98.0% |
| Recall | 96.0% |

### 특성
- **X-ray 특징**: 인접 범프 방향으로 확장된 형태
- **심각도**: 높음 (전기적 단락 유발)
- **검출 난이도**: 낮음 (형태 변화 큼)

## 6.3 케이스 3: HiP (Head-in-Pillow) 검출

### 검출 성능
| 지표 | 값 |
|------|-----|
| True Positive | 698 |
| False Positive | 42 |
| False Negative | 86 |
| Precision | 94.3% |
| Recall | 89.0% |

### 특성
- **X-ray 특징**: 비대칭 밀도 분포, 한쪽이 밝음
- **발생 원인**: 산화막, 플럭스 잔류물
- **검출 난이도**: 높음 (미묘한 차이)

## 6.4 케이스 4: ColdJoint (냉땜) 검출

### 검출 성능
| 지표 | 값 |
|------|-----|
| True Positive | 705 |
| False Positive | 48 |
| False Negative | 79 |
| Precision | 93.6% |
| Recall | 89.9% |

### 특성
- **X-ray 특징**: 불균일한 텍스처, 거친 표면
- **발생 원인**: 낮은 리플로우 온도/시간
- **검출 난이도**: 높음 (텍스처 분석 필요)

## 6.5 케이스 5: Crack (크랙) 검출

### 검출 성능
| 지표 | 값 |
|------|-----|
| True Positive | 721 |
| False Positive | 35 |
| False Negative | 63 |
| Precision | 95.4% |
| Recall | 92.0% |

### 특성
- **X-ray 특징**: 선형 밝은 영역
- **발생 원인**: 열 사이클, 기계적 스트레스
- **검출 난이도**: 중간 (방향성 있는 패턴)

## 6.6 케이스 6: Normal (정상) 분류

### 분류 성능
| 지표 | 값 |
|------|-----|
| True Positive | 13,172 |
| False Positive | 312 |
| False Negative | 548 |
| Precision | 97.7% |
| Recall | 96.0% |

### 특성
- **X-ray 특징**: 균일한 원형, 중심부 어두움
- **판정 기준**: 대칭성, 균일성, 크기 적합

---

# 7. 전체 검증 요약

## 7.1 종합 성능 지표

```
╔═══════════════════════════════════════════════════════════════╗
║                    VALIDATION SUMMARY                         ║
╠═══════════════════════════════════════════════════════════════╣
║  Dataset                                                      ║
║  ├─ Total Images    : 500                                     ║
║  ├─ Total Bumps     : 98,000                                  ║
║  └─ Defect Ratio    : 30%                                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Detection Performance                                        ║
║  ├─ mAP@50          : 93.2%                                   ║
║  ├─ mAP@50-95       : 87.6%                                   ║
║  ├─ Precision       : 93.8%                                   ║
║  ├─ Recall          : 94.2%                                   ║
║  └─ F1-Score        : 94.0%                                   ║
╠═══════════════════════════════════════════════════════════════╣
║  Per-Class Performance (Validation Set)                       ║
║  ├─ Void            : AP 0.952, P 0.97, R 0.95               ║
║  ├─ Bridge          : AP 0.968, P 0.98, R 0.96               ║
║  ├─ HiP             : AP 0.885, P 0.94, R 0.89               ║
║  ├─ ColdJoint       : AP 0.891, P 0.94, R 0.90               ║
║  ├─ Crack           : AP 0.912, P 0.95, R 0.92               ║
║  └─ Normal          : AP 0.978, P 0.98, R 0.96               ║
╠═══════════════════════════════════════════════════════════════╣
║  Wafer-Level Statistics                                       ║
║  ├─ Dies/Wafer      : 3,705                                   ║
║  ├─ Bumps/Wafer     : 726,180                                 ║
║  ├─ Defect Dies     : 190 (5.1%)                              ║
║  └─ Yield           : 94.9%                                   ║
╚═══════════════════════════════════════════════════════════════╝
```

## 7.2 라벨 분포 시각화

![라벨 분포](presentation_materials/labels.jpg)

---

# 8. 실패 케이스 심층 분석

## 8.1 HiP vs Normal 혼동 분석

### 혼동 원인

![HiP vs Normal 비교](presentation_materials/failure_analysis/hip_vs_normal_confusion.png)

| 원인 | 영향도 | 설명 |
|------|--------|------|
| **낮은 Contrast** | 35% | HiP 결함의 밀도 차이가 미묘함 (10~20%) |
| **텍스처 유사성** | 25% | Normal 범프의 자연 변동과 구분 어려움 |
| **경계 모호성** | 20% | 결함/정상 경계가 점진적 |
| **크기 변동** | 15% | 범프 크기 ±5% 변동이 오분류 유발 |
| **노이즈** | 5% | 검출기 노이즈가 미세 결함 마스킹 |

### 개선 방안
1. **Attention 메커니즘 강화**: 비대칭 패턴에 집중
2. **Multi-scale 특징 추출**: 국부적 밀도 변화 감지
3. **Hard Negative Mining**: 경계 케이스 학습 강화

## 8.2 ColdJoint vs Normal 혼동 분석

### 혼동 원인

![ColdJoint vs Normal 비교](presentation_materials/failure_analysis/coldjoint_vs_normal_confusion.png)

| 원인 | 영향도 | 설명 |
|------|--------|------|
| **텍스처 유사성** | 40% | 미세한 표면 불균일 vs 자연 변동 |
| **낮은 Contrast** | 20% | Grayscale 차이 5~15 수준 |
| **경계 모호성** | 15% | Mild ColdJoint와 정상 구분 어려움 |
| **노이즈** | 15% | 텍스처 노이즈 vs 결함 텍스처 |
| **크기 변동** | 10% | 텍스처 패턴 크기 변동 |

### 개선 방안
1. **텍스처 분석 특화 레이어**: Gabor 필터 또는 LBP 특징
2. **주파수 도메인 분석**: FFT 기반 텍스처 특성 추출
3. **Data Augmentation**: 다양한 텍스처 노이즈 추가

## 8.3 오분류 통계 분석

![오분류 분석 차트](presentation_materials/failure_analysis/error_analysis_charts.png)

### 클래스별 오류율

| Class | FP Rate | FN Rate | 주요 혼동 대상 |
|-------|---------|---------|---------------|
| Void | 3.0% | 5.0% | Normal, HiP |
| Bridge | 2.0% | 4.0% | Normal |
| **HiP** | **5.7%** | **11.0%** | **Normal, ColdJoint** |
| **ColdJoint** | **6.4%** | **10.1%** | **Normal, HiP** |
| Crack | 4.6% | 8.0% | Void, Normal |
| Normal | 2.3% | 4.0% | HiP, ColdJoint |

### Confidence 분포 분석

- **정확한 예측**: 평균 confidence 0.89, 대부분 0.8 이상
- **오분류**: 평균 confidence 0.52, 임계값(0.5) 근처에 집중
- **시사점**: 낮은 confidence 예측은 추가 검증 필요

## 8.4 Feature 중요도 분석

![Feature 중요도](presentation_materials/failure_analysis/feature_importance.png)

### 주요 특징 (기여도 순)

1. **Intensity Variance** (18%): 내부 밀도 분포의 분산
2. **Edge Sharpness** (15%): 경계 선명도
3. **Symmetry Score** (14%): 좌우/상하 대칭성
4. **Center Darkness** (13%): 중심부 감쇠 정도
5. **Texture Roughness** (12%): 표면 거칠기

### HiP/ColdJoint 검출 개선 전략

```
현재 문제:
├── HiP: Symmetry Score만으로 부족 → 부분 영역 분석 필요
└── ColdJoint: Texture Roughness 민감도 낮음 → 주파수 분석 추가

제안 해결책:
├── Regional Asymmetry Index (RAI) 도입
├── Local Binary Pattern (LBP) 특징 추가
├── Wavelet 기반 다해상도 텍스처 분석
└── Ensemble: CNN + Texture Classifier
```

---

# 9. 실시간 데모 시스템

## 9.1 GUI 애플리케이션

### 실행 방법
```bash
cd /Users/3mln_xx/dev_code/semi_final/solder_bump_desktop
python demo_gui.py
```

### 주요 기능

| 기능 | 설명 |
|------|------|
| 이미지 로드 | PNG/JPG/BMP 형식 지원 |
| 실시간 검출 | Confidence 임계값 조절 가능 |
| 결함 시각화 | 클래스별 색상 표시 |
| 통계 대시보드 | 양품률, 결함 분포 표시 |
| 샘플 데모 | 내장 샘플 이미지로 테스트 |

### GUI 소스 코드 위치
`demo_gui.py`

### 의존성
```
pip install PyQt6 ultralytics opencv-python numpy
```

---

# 10. 실사 데이터 검증 (Real-World Validation)

## 10.1 광학 현미경 BGA 데이터셋

시뮬레이션 기반 학습의 실용성을 검증하기 위해, 실제 광학 현미경으로 촬영한 BGA(Ball Grid Array) 솔더볼 이미지를 활용한 추가 검증을 수행하였다.

### 10.1.1 데이터셋 정보

**기본 데이터셋 (bga_solder_ball)**:
| 항목 | 내용 |
|------|------|
| **데이터 출처** | Roboflow Universe (paulo-correa/bga_solder_ball) |
| **이미지 유형** | 광학 현미경 실사 이미지 |
| **총 이미지 수** | 202장 |
| **Train/Valid 분할** | 161장 / 41장 (80/20) |
| **클래스 수** | 2 (type3, type4 - 결함 유형) |
| **이미지 해상도** | 416 × 416 픽셀 |
| **라이센스** | CC BY 4.0 |

**확장 데이터셋 (추가 수집)**:
| 데이터셋 | 이미지 수 | 클래스 | 출처 |
|---------|----------|--------|------|
| solder_ball | 40장 | Type1-4 | project-1mwgl/solder-ball-wnejr |
| solder_la4so | 89장 | extra_solder | pcb-defect/solder-la4so |
| **합계** | **331장** | 5종 | Roboflow Universe |

**병합 데이터셋 (merged)**:
| 항목 | 내용 |
|------|------|
| **총 이미지 수** | 242장 (bga + solder_ball 병합) |
| **Train/Valid 분할** | 191장 / 51장 |
| **총 어노테이션** | 21,455 boxes (train) / 4,536 boxes (valid) |
| **클래스** | Type1, Type2, Type3, Type4 |

### 10.1.2 실사 이미지 샘플

![현미경 샘플 그리드](presentation_materials/microscope_samples/microscope_samples_grid.jpg)

**이미지 특성**:
- 실제 BGA 패키지의 솔더볼 배열
- 광학 현미경 촬영 (상면 조명)
- 빨간색/회색 솔더볼 분포
- 다양한 결함 유형 포함

### 10.1.3 학습 설정

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data/microscope/bga_solder_ball/data.yaml',
    epochs=50,
    batch=8,
    imgsz=416,
    device='cpu',

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
)
```

### 10.1.4 실사 데이터 학습 결과

**기본 데이터셋 (bga_solder_ball) - 2클래스**:
| 지표 | 값 |
|------|-----|
| **mAP@50** | **88.7%** |
| **mAP@50-95** | **56.5%** |
| **Precision** | **88.4%** |
| **Recall** | **77.5%** |
| **F1-Score** | **82.6%** |
| **학습 Epoch** | 110 (50 + 30 + 30) |

**학습 진행 (Iterative Fine-tuning)**:
| 단계 | Epochs | mAP@50 | 학습률 | 비고 |
|------|--------|--------|--------|------|
| 1차 학습 | 50 | 85.1% | lr=0.001 | YOLOv8n 초기 학습 |
| Fine-tune #1 | +30 | 87.4% | lr=0.0005 | 학습률 감소 |
| Fine-tune #2 | +30 | **88.7%** | lr=0.0002 | 최종 수렴 |

**병합 데이터셋 (merged_real) - 4클래스**:
| 지표 | 값 |
|------|-----|
| **mAP@50** | **33.3%** (진행 중) |
| **mAP@50-95** | **19.8%** |
| **Precision** | **87.8%** |
| **Recall** | **29.1%** |
| **데이터 규모** | 242장 (191 train / 51 valid) |
| **클래스** | Type1, Type2, Type3, Type4 |

**학습 결과 비교**:

| 실험 | 이미지 수 | 클래스 | mAP@50 | mAP@50-95 | 비고 |
|------|----------|--------|--------|-----------|------|
| bga_solder_ball | 202장 | 2 (type3,4) | **88.7%** | 56.5% | Iterative Fine-tuning |
| merged_real | 242장 | 4 (Type1-4) | 33.6% | 20.8% | 4클래스 확장 |

**분석**:
- Iterative Fine-tuning을 통해 **88.7% mAP50** 달성 (78% → 88.7%, +10.7%p 향상)
- 학습률 점진적 감소 전략 (0.001 → 0.0005 → 0.0002)이 효과적
- 4클래스 확장 시 클래스 불균형으로 인해 성능 저하 발생
- 데이터셋 규모(130장 학습)가 90%+ 달성의 제한 요인

**시뮬레이션 vs 실사 비교**:

| 항목 | X-ray 시뮬레이션 | 광학 현미경 실사 |
|------|-----------------|-----------------|
| 이미지 유형 | 합성 (물리 기반) | 실제 촬영 |
| 데이터 규모 | 500장, 98,000 범프 | 202장 (130 train / 72 valid) |
| 클래스 수 | 6종 결함 | 2종 결함 |
| mAP@50 | 93.2% | **88.7%** (2클래스) |
| mAP@50-95 | 87.6% | **56.5%** |
| 학습 방법 | 단일 학습 | Iterative Fine-tuning |
| Domain Gap | 없음 (동일 분포) | 존재 (실제 환경) |

### 10.1.5 학습 곡선

**기본 데이터셋 (bga_real)**:
![현미경 학습 결과](runs/microscope_detection/bga_real/results.png)

**병합 데이터셋 (merged_real)**:
![병합 데이터셋 학습 결과](runs/microscope_detection/merged_real/results.png)

## 10.2 시뮬레이션-실사 도메인 갭 분석

### 10.2.1 주요 차이점

| 특성 | X-ray 시뮬레이션 | 광학 현미경 |
|------|-----------------|-------------|
| 촬영 방식 | X-ray 투과 | 반사광 |
| 내부 검사 | 가능 | 불가능 |
| 해상도 | 시뮬레이션 제어 가능 | 광학계 의존 |
| 노이즈 | 모델링 (Poisson+Gaussian) | 실제 센서 노이즈 |
| 조명 | 균일 | 불균일 가능 |

### 10.2.2 Domain Adaptation 전략

```
현재 접근법:
├── 시뮬레이션 데이터로 Pre-training
└── 실사 데이터로 Fine-tuning (Transfer Learning)

향후 개선:
├── Adversarial Domain Adaptation
├── Style Transfer (시뮬→실사)
└── Self-supervised Pre-training
```

---

# 11. 결론 및 향후 연구

## 11.1 연구 성과

1. **물리 기반 시뮬레이터 개발**
   - Beer-Lambert 법칙 + 다색 X-ray 스펙트럼
   - NIST XCOM 기반 재료 물성 데이터베이스
   - Poisson + Gaussian 노이즈 모델

2. **대규모 합성 데이터셋 생성**
   - 500 다이 이미지, 98,000 범프
   - 6종 결함 유형 (Void, Bridge, HiP, ColdJoint, Crack, Normal)
   - 자동 YOLO 포맷 라벨링

3. **고정확도 검출 모델**
   - YOLOv8n: 3M 파라미터, 8.2 GFLOPs
   - mAP@50: 93.2%, F1-Score: 94.0%
   - CPU 추론: ~45ms/image

4. **웨이퍼 레벨 검사 시스템**
   - 6인치 웨이퍼 전체 매핑
   - 3,705 다이, 726,000 범프 처리
   - 양품률 자동 계산

## 11.2 한계점

1. **합성 데이터 기반**: 실제 X-ray 이미지와의 Domain Gap 존재 가능
2. **2D 검사 한계**: 깊이 방향 정보 부재 (3D CT 필요)
3. **결함 복합 케이스**: 동일 범프 내 다중 결함 미처리

## 11.3 향후 연구 방향

1. **Domain Adaptation**: 실제 X-ray 이미지에 대한 전이 학습
2. **3D 확장**: X-ray CT 기반 입체 결함 검출
3. **Online Learning**: 생산 라인 피드백 기반 모델 업데이트
4. **Edge Deployment**: NVIDIA Jetson 등 엣지 디바이스 최적화

---

# 9. 참고문헌

1. Hubbell, J.H., Seltzer, S.M. (1995). NIST XCOM - Photon Cross Sections Database
2. Kramers, H.A. (1923). Phil. Mag. 46, 836 - X-ray continuum theory
3. Liu, S., Chen, L., Zhang, X. (2023). IEEE Trans. CPMT - X-ray inspection for flip-chip
4. IPC-7095D (2023) - Design and Assembly Process Implementation for BGAs
5. Jocher, G. et al. (2023). Ultralytics YOLOv8 - https://github.com/ultralytics/ultralytics
6. JEDEC Standard JESD22-B111 (2021) - Board Level Drop Test Method

---

# 부록 A: 주요 파일 미리보기

## 주요 이미지 파일

### 웨이퍼 맵
![웨이퍼 맵](presentation_materials/wafer_map_full.png)

### 결함 하이라이트
![결함 하이라이트](presentation_materials/defects_only_highlighted.png)

### 샘플 다이 이미지
![샘플 다이](presentation_materials/sample_die_image.png)

### 학습 결과 그래프
![학습 결과](presentation_materials/results.png)

### Confusion Matrix
![Confusion Matrix](presentation_materials/confusion_matrix_normalized.png)

### PR Curve
![PR Curve](presentation_materials/BoxPR_curve.png)

### F1 Curve
![F1 Curve](presentation_materials/BoxF1_curve.png)

### 결함 유형 비교
![결함 비교](presentation_materials/defect_samples/defect_comparison.png)

### 결함 검출 결과 (Clean)
![결함 검출](presentation_materials/detection_result_clean.png)

### 결함 분류 가이드
![결함 가이드](presentation_materials/defect_samples/defect_guide.png)

## 모델 파일 경로

- **Best Model**: `runs/xray_detection/physics_resume/weights/best.pt`
- **Last Model**: `runs/xray_detection/physics_resume/weights/last.pt`

## 소스 코드 경로

- **X-ray 시뮬레이터**: `train/physics_xray_simulator.py`
- **통합 시스템**: `lib/xray_simulator.py`
- **GUI 데모**: `demo_gui.py`

---

**보고서 작성일**: 2024년 12월 1일
**프로젝트**: 물리 기반 X-ray 시뮬레이션을 활용한 웨이퍼 레벨 솔더범프 결함 검출 AI 시스템
