# 솔더 범프 결함 검출 시스템 - TODO

## Phase 1: 프로젝트 초기화 및 구조 설계
- [x] 프로젝트 디렉토리 구조 생성
- [x] README.md 작성
- [x] requirements.txt 작성

## Phase 2: YOLOv8 모델 학습 코드 작성
- [x] 현미경 데이터셋 다운로드 스크립트
- [x] X-ray 데이터셋 준비 스크립트
- [x] X-ray 물리 시뮬레이션 기반 데이터 생성기 통합 (lib/xray_simulator.py)
- [x] YOLOv8 형식 데이터 생성기 (train/generate_yolo_dataset.py)
- [x] 현미경 모델 학습 스크립트
- [x] X-ray 모델 학습 스크립트 (MPS 지원 추가)

## Phase 3: PyQt6 GUI 애플리케이션 개발
- [x] 메인 윈도우 UI 설계
- [x] 검사 유형 선택 (현미경/X-ray)
- [x] 단일 이미지 업로드 기능
- [x] 폴더 일괄 처리 기능
- [x] YOLOv8 모델 로더
- [x] 실시간 검출 결과 표시
- [x] 이미지 뷰어 (결함 바운딩 박스)

## Phase 4: 웨이퍼맵 시각화 및 리포트 기능
- [x] 웨이퍼맵 2D 히트맵 생성
- [x] 결함 분포 시각화
- [x] 논문용 평가 그래프
  - [x] Confusion Matrix
  - [x] Precision-Recall Curve
  - [x] F1-Confidence Curve
  - [x] Training Loss Curve
  - [x] 클래스별 검출 분포
  - [x] 신뢰도 분포 히스토그램
- [x] 통계 리포트 생성
- [x] PDF 리포트 생성
- [x] Excel/CSV 데이터 내보내기
- [x] LaTeX 테이블 생성

## Phase 5: 테스트 및 최종 결과 전달
- [ ] 샘플 이미지로 시연
- [ ] 폴더 일괄 처리 테스트
- [ ] 성능 벤치마크
- [x] 사용자 가이드 작성 (USER_GUIDE.md)

---

## 사용 방법

### 1. X-ray 데이터셋 생성
```bash
cd train
python download_xray.py --num_images 500
```

### 2. X-ray 모델 학습
```bash
python train_xray.py
```

### 3. 앱 실행
```bash
cd app
python main.py
```
