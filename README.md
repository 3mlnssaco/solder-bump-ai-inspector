# 솔더 범프 결함 검출 시스템 (PyQt6 Desktop)

웨이퍼 레벨에서 모든 솔더 범프를 검사하고 결함을 검출하는 데스크톱 애플리케이션입니다.

## 프로젝트 구조

```
solder_bump_desktop/
├── train/                      # 모델 학습 코드
│   ├── download_microscope.py  # 현미경 데이터셋 다운로드
│   ├── download_xray.py        # X-ray 데이터셋 다운로드
│   ├── train_microscope.py     # 현미경 모델 학습
│   ├── train_xray.py           # X-ray 모델 학습
│   └── evaluate.py             # 모델 평가
├── app/                        # PyQt6 애플리케이션
│   ├── main.py                 # 메인 애플리케이션
│   ├── ui/                     # UI 컴포넌트
│   │   ├── main_window.py
│   │   ├── upload_widget.py
│   │   └── result_widget.py
│   ├── models/                 # 모델 로더
│   │   └── detector.py
│   └── utils/                  # 유틸리티
│       ├── wafer_map.py
│       └── report.py
├── data/                       # 데이터셋 저장 위치
├── models/                     # 학습된 모델 저장 위치
└── requirements.txt
```

## 주요 기능

### 1. 검사 유형
- **현미경 검사**: Type1, Type2, Type3, Type4 분류
- **X-ray 검사**: Void, Bridge, HiP, ColdJoint, Crack 검출

### 2. 핵심 기능
- 웨이퍼 전체 이미지 업로드
- YOLOv8 기반 실시간 결함 검출
- 웨이퍼맵 시각화 (결함 분포 2D 맵)
- 검사 리포트 생성 (PDF/Excel)

## 설치 방법

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

### 1. 모델 학습 (선택사항)

```bash
# 현미경 모델 학습
cd train
python download_microscope.py
python train_microscope.py

# X-ray 모델 학습
python download_xray.py
python train_xray.py
```

### 2. 애플리케이션 실행

```bash
cd app
python main.py
```

## 기술 스택

- **GUI**: PyQt6
- **모델**: YOLOv8 (Ultralytics)
- **시각화**: Matplotlib, Seaborn
- **리포트**: ReportLab (PDF), OpenPyXL (Excel)
- **이미지 처리**: OpenCV, Pillow

## 데이터셋

### 현미경 검사
- Roboflow BGA Solder Ball Dataset
- Roboflow Solder Ball Dataset

### X-ray 검사
- 물리 기반 X-ray 시뮬레이션 데이터 (10,000장)
- 7가지 결함 유형 포함

## 라이선스

MIT License
