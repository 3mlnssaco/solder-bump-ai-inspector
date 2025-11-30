# Solder Bump AI Inspector

YOLOv8 기반 반도체 웨이퍼 레벨 패키지(WLP) 솔더 범프 결함 검출 시스템

## 주요 기능

- **현미경 이미지 분석**: BGA 솔더볼 결함 검출 (Type1~4)
- **X-ray 이미지 분석**: Void, Bridge, HiP, ColdJoint, Crack 검출
- **GUI 데모**: PyQt6 기반 데스크톱 애플리케이션
- **웨이퍼맵 시각화**: 결함 분포 2D 맵 생성

## 학습된 모델 성능

| 모델 | 용도 | mAP@50 | mAP@50-95 |
|------|------|--------|-----------|
| `microscope_bga_best.pt` | BGA 솔더볼 (2 classes) | **78.0%** | 47.7% |
| `microscope_merged_best.pt` | 통합 현미경 (4 classes) | 33.6% | 20.8% |
| `xray_physics_best.pt` | X-ray 물리 시뮬레이션 | - | - |

## 프로젝트 구조

```
solder-bump-ai-inspector/
├── models/                 # 학습된 모델 가중치
├── app/                    # PyQt6 애플리케이션
├── train/                  # 학습 스크립트
├── lib/                    # 라이브러리 (X-ray 시뮬레이터)
├── presentation_materials/ # 발표 자료
├── demo_gui.py            # GUI 데모
└── REPORT.md              # 상세 연구 보고서
```

## 설치

```bash
pip install -r requirements.txt
```

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

## 기술 스택

- **Detection**: YOLOv8 (Ultralytics)
- **GUI**: PyQt6
- **Data**: Roboflow Universe

## 데이터셋

- [BGA Solder Ball Dataset](https://universe.roboflow.com/project-1mwgl/solder-ball-wnejr) (Roboflow)
- 물리 기반 X-ray 시뮬레이션 데이터

## License

MIT License
