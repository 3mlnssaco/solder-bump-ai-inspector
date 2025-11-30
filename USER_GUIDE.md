# 솔더 범프 결함 검출 시스템 - 사용자 가이드

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [설치 방법](#설치-방법)
3. [모델 학습](#모델-학습)
4. [애플리케이션 사용법](#애플리케이션-사용법)
5. [리포트 생성](#리포트-생성)
6. [문제 해결](#문제-해결)

---

## 시스템 개요

본 시스템은 웨이퍼 레벨에서 솔더 범프의 결함을 자동으로 검출하는 PyQt6 기반 데스크톱 애플리케이션입니다.

### 주요 기능

- **두 가지 검사 유형**
  - 현미경 검사: Type1, Type2, Type3, Type4 분류
  - X-ray 검사: Void, Bridge, HiP, ColdJoint, Crack 검출

- **유연한 입력 방식**
  - 단일 이미지 업로드
  - 폴더 일괄 처리 (배치 검사)

- **논문용 평가 기능**
  - Confusion Matrix
  - Precision-Recall Curve
  - F1-Confidence Curve
  - Training Loss Curve
  - 웨이퍼맵 시각화

- **다양한 리포트 형식**
  - PDF 리포트
  - Excel 데이터
  - LaTeX 테이블

---

## 설치 방법

### 1. 시스템 요구사항

- **OS**: Windows 10/11, macOS, Linux (Ubuntu 20.04+)
- **Python**: 3.8 이상
- **GPU**: CUDA 지원 GPU 권장 (CPU도 가능하지만 느림)
- **RAM**: 8GB 이상 권장

### 2. 의존성 설치

```bash
# 프로젝트 디렉토리로 이동
cd solder_bump_desktop

# 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. GPU 설정 (선택사항)

CUDA를 사용하려면 PyTorch GPU 버전을 설치하세요:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 모델 학습

### 1. 현미경 모델 학습

```bash
cd train

# 1단계: 데이터셋 다운로드
python download_microscope.py

# 2단계: 모델 학습
python train_microscope.py
```

**학습 설정:**
- 모델: YOLOv8n (nano)
- Epochs: 100
- Batch Size: 16
- Image Size: 640×640

**출력:**
- 학습된 모델: `../models/microscope_best.pt`
- 학습 로그: `microscope_detection/train/`

### 2. X-ray 모델 학습

```bash
cd train

# 1단계: 데이터셋 준비
python download_xray.py

# 2단계: 실제 X-ray 이미지 추가
# data/xray/train/images/ 폴더에 이미지 복사
# data/xray/train/labels/ 폴더에 YOLO 형식 라벨 복사

# 3단계: 모델 학습
python train_xray.py
```

**YOLO 라벨 형식:**
```
# example.txt
<class_id> <x_center> <y_center> <width> <height>

# 예시 (정규화된 좌표 0-1)
0 0.5 0.5 0.1 0.1  # Void at center
1 0.3 0.7 0.15 0.08  # Bridge at bottom-left
```

### 3. 학습 모니터링

TensorBoard로 학습 과정을 모니터링할 수 있습니다:

```bash
# 현미경 모델
tensorboard --logdir=microscope_detection/train

# X-ray 모델
tensorboard --logdir=xray_detection/train
```

브라우저에서 `http://localhost:6006` 접속

---

## 애플리케이션 사용법

### 1. 애플리케이션 실행

```bash
cd app
python main.py
```

### 2. 모델 로드

1. **검사 유형 선택**: Microscope 또는 X-ray
2. **Load Model 버튼 클릭**
3. 학습된 모델 파일 선택 (`.pt` 파일)
   - 현미경: `models/microscope_best.pt`
   - X-ray: `models/xray_best.pt`

### 3. 이미지 입력

#### 방법 1: 단일 이미지

1. **Select Single Image 버튼 클릭**
2. 검사할 이미지 선택 (PNG, JPG, JPEG, BMP)

#### 방법 2: 폴더 일괄 처리 (권장)

1. **Select Folder (Batch) 버튼 클릭**
2. 이미지가 들어있는 폴더 선택
3. 폴더 내 모든 이미지가 자동으로 검출됨

### 4. 검출 설정

- **Confidence**: 신뢰도 임계값 (기본: 0.25)
  - 높을수록 확실한 검출만 표시
  - 낮을수록 더 많은 검출 (False Positive 증가)

- **IOU**: Intersection over Union 임계값 (기본: 0.45)
  - NMS (Non-Maximum Suppression)에 사용
  - 중복 검출 제거

### 5. 검출 실행

1. **Run Detection 버튼 클릭**
2. 진행률 바에서 진행 상황 확인
3. 완료 후 결과 탭에서 통계 확인

### 6. 결과 확인

- **Log 탭**: 실시간 진행 상황 및 로그
- **Results 탭**: 검출 통계 요약
- **Visualization 탭**: 웨이퍼맵 및 그래프

---

## 리포트 생성

### 1. PDF 리포트

1. **Generate PDF Report 버튼 클릭**
2. 저장 위치 선택
3. PDF 파일 생성 완료

**포함 내용:**
- 요약 통계
- 클래스별 검출 수
- 시각화 그래프

### 2. Excel 리포트

1. **Generate Excel Report 버튼 클릭**
2. 저장 위치 선택
3. Excel 파일 생성 완료

**포함 시트:**
- Summary: 요약 통계
- Class Distribution: 클래스별 분포
- Detections: 상세 검출 결과

### 3. LaTeX 테이블

1. **Generate LaTeX Table 버튼 클릭**
2. 저장 위치 선택
3. `.tex` 파일 생성 완료

**사용 방법:**
```latex
\documentclass{article}
\begin{document}

\input{defect_table.tex}

\end{document}
```

---

## 문제 해결

### Q1: 모델 로드 실패

**증상**: "Failed to load model" 오류

**해결:**
1. 모델 파일 경로 확인
2. 올바른 검사 유형 선택 확인
3. PyTorch 버전 확인 (`pip list | grep torch`)

### Q2: GPU 메모리 부족

**증상**: "CUDA out of memory" 오류

**해결:**
1. Batch Size 줄이기 (학습 시)
2. 이미지 크기 줄이기
3. CPU 모드로 전환

### Q3: 검출 속도가 느림

**해결:**
1. GPU 사용 확인
2. 더 작은 모델 사용 (yolov8n)
3. 이미지 크기 줄이기

### Q4: 검출 정확도가 낮음

**해결:**
1. 신뢰도 임계값 조정
2. 더 많은 데이터로 재학습
3. 더 큰 모델 사용 (yolov8m, yolov8l)
4. 데이터 증강 강화

### Q5: 라벨 형식 오류

**증상**: "Invalid label format" 오류

**해결:**
1. YOLO 형식 확인 (5개 값: class x y w h)
2. 좌표 정규화 확인 (0-1 범위)
3. 클래스 ID 범위 확인

---

## 성능 벤치마크

### 하드웨어별 추론 속도

| 하드웨어 | 모델 | 이미지 크기 | FPS |
|---------|------|-----------|-----|
| RTX 3090 | YOLOv8n | 640×640 | ~150 |
| RTX 3090 | YOLOv8m | 640×640 | ~80 |
| RTX 3060 | YOLOv8n | 640×640 | ~100 |
| CPU (i7) | YOLOv8n | 640×640 | ~10 |

### 모델 크기 비교

| 모델 | 파라미터 | 크기 | mAP@0.5 |
|------|---------|------|---------|
| YOLOv8n | 3.2M | 6.2MB | ~0.85 |
| YOLOv8s | 11.2M | 21.5MB | ~0.88 |
| YOLOv8m | 25.9M | 49.7MB | ~0.91 |
| YOLOv8l | 43.7M | 83.7MB | ~0.93 |

---

## 추가 리소스

- **YOLOv8 문서**: https://docs.ultralytics.com/
- **PyQt6 문서**: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- **Roboflow Universe**: https://universe.roboflow.com/

---

## 라이선스

MIT License

## 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.
