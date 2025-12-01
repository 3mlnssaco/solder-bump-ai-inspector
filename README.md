# Solder Bump AI Inspector

YOLOv8 기반 반도체 웨이퍼 레벨 패키지(WLP) 솔더 범프 결함 검출 시스템

![Detection Result](./presentation_materials/detection_result_clean.png)

## 주요 기능

- **현미경 이미지 분석**: BGA 솔더볼 결함 검출 (Type1~4)
- **X-ray 이미지 분석**: Void, Bridge, HiP, ColdJoint, Crack 검출
- **GUI 데모**: PyQt6 기반 데스크톱 애플리케이션
- **웨이퍼맵 시각화**: 결함 분포 2D 맵 생성

## 학습 결과

### 모델 성능

| 모델 | 용도 | mAP@50 | mAP@50-95 | 비고 |
|------|------|--------|-----------|------|
| [`microscope_bga_best.pt`](./models/microscope_bga_best.pt) | BGA 솔더볼 (2 classes) | **88.7%** | 56.5% | YOLOv8n, 110 epochs |
| [`microscope_merged_best.pt`](./models/microscope_merged_best.pt) | 통합 현미경 (4 classes) | 33.6% | 20.8% | |
| [`xray_physics_best.pt`](./models/xray_physics_best.pt) | X-ray 물리 시뮬레이션 (6 classes) | 93.2% | 87.6% | |

### 학습 진행 (BGA 모델)

| 단계 | Epochs | mAP@50 | 학습률 |
|------|--------|--------|--------|
| 1차 학습 | 50 | 85.1% | lr=0.001 |
| Fine-tune #1 | +30 | 87.4% | lr=0.0005 |
| Fine-tune #2 | +30 | **88.7%** | lr=0.0002 |

### 학습 곡선

<p align="center">
  <img src="./presentation_materials/results.png" width="600"/>
</p>

### 혼동 행렬

<p align="center">
  <img src="./presentation_materials/confusion_matrix_normalized.png" width="500"/>
</p>

## 프로젝트 구조

```
solder-bump-ai-inspector/
├── models/                 # 학습된 모델 가중치
│   ├── microscope_bga_best.pt
│   ├── microscope_merged_best.pt
│   └── xray_physics_best.pt
├── app/                    # PyQt6 애플리케이션
├── train/                  # 학습 스크립트
├── lib/                    # 라이브러리 (X-ray 시뮬레이터)
├── presentation_materials/ # 발표 자료 및 시각화
├── demo_gui.py            # GUI 데모
├── evaluate_model.py      # 모델 평가
└── REPORT.md              # 상세 연구 보고서
```

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 필수 패키지
- Python 3.8+
- PyTorch 2.0+
- Ultralytics (YOLOv8)
- PyQt6
- OpenCV
- NumPy

## 사용법

### GUI 데모 실행
```bash
python demo_gui.py
```

### 모델 학습 (선택)
```bash
cd train
python download_microscope.py  # 데이터셋 다운로드
python train_microscope.py     # 현미경 모델 학습
```

### 모델 평가
```bash
python evaluate_model.py
```

## 결함 유형

### X-ray 검출 대상
| 결함 | 설명 |
|------|------|
| Void | 솔더 내부 기공 |
| Bridge | 인접 범프 간 단락 |
| HiP (Head in Pillow) | 불완전 접합 |
| ColdJoint | 냉납 (불량 접합) |
| Crack | 균열 |
| Normal | 정상 |

### 현미경 검출 대상
- **Type1**: 정상 솔더볼
- **Type2**: 결함 솔더볼
- **Type3**: 크랙/손상
- **Type4**: 브릿지/쇼트

## 기술 스택

- **Detection**: YOLOv8 (Ultralytics)
- **GUI**: PyQt6
- **Simulation**: 물리 기반 X-ray 시뮬레이션
- **Data**: Roboflow Universe

## 데이터셋

- [BGA Solder Ball Dataset](https://universe.roboflow.com/project-1mwgl/solder-ball-wnejr) (Roboflow)
- 물리 기반 X-ray 시뮬레이션 데이터 (자체 생성)

## 샘플 이미지

### 현미경 샘플
<p align="center">
  <img src="./presentation_materials/microscope_samples/microscope_samples_grid.jpg" width="600"/>
</p>

### 웨이퍼 맵
<p align="center">
  <img src="./presentation_materials/wafer_map_full.png" width="500"/>
</p>

## 주요 파일

| 파일 | 설명 |
|------|------|
| [`demo_gui.py`](./demo_gui.py) | GUI 데모 애플리케이션 |
| [`evaluate_model.py`](./evaluate_model.py) | 모델 평가 스크립트 |
| [`REPORT.md`](./REPORT.md) | 상세 연구 보고서 |
| [`requirements.txt`](./requirements.txt) | Python 의존성 목록 |

## 참고 문헌

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)

## License

MIT License
