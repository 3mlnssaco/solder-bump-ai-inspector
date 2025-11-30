"""
현미경 이미지 YOLOv8 모델 학습 스크립트
"""

import os
from ultralytics import YOLO
import torch

# 학습 설정
DATA_YAML = "../data/microscope/merged/data.yaml"
MODEL_SIZE = "yolov8n"  # n, s, m, l, x 중 선택
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "microscope_detection"
OUTPUT_DIR = "../models"

def check_data():
    """데이터셋 확인"""
    print("=" * 60)
    print("데이터셋 확인 중...")
    print("=" * 60)
    
    if not os.path.exists(DATA_YAML):
        print(f"\n✗ 데이터셋을 찾을 수 없습니다: {DATA_YAML}")
        print("먼저 download_microscope.py를 실행하세요.")
        return False
    
    print(f"\n✓ 데이터셋 발견: {DATA_YAML}")
    return True

def train_model():
    """YOLOv8 모델 학습"""
    print("\n" + "=" * 60)
    print("YOLOv8 모델 학습 시작")
    print("=" * 60)
    
    print(f"\n학습 설정:")
    print(f"  - 모델: {MODEL_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Image Size: {IMG_SIZE}")
    print(f"  - Device: {DEVICE}")
    
    # YOLOv8 모델 로드
    model = YOLO(f"{MODEL_SIZE}.pt")
    
    # 학습 시작
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name="train",
        patience=20,  # Early stopping
        save=True,
        save_period=10,  # 10 epoch마다 체크포인트 저장
        
        # 데이터 증강
        augment=True,
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=10.0,  # Rotation
        translate=0.1, # Translation
        scale=0.5,     # Scaling
        shear=0.0,     # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,    # Flip up-down
        fliplr=0.5,    # Flip left-right
        mosaic=1.0,    # Mosaic augmentation
        mixup=0.0,     # Mixup augmentation
        
        # 최적화
        optimizer="AdamW",
        lr0=0.01,      # Initial learning rate
        lrf=0.01,      # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 기타
        verbose=True,
        plots=True,
    )
    
    print("\n" + "=" * 60)
    print("✓ 학습 완료!")
    print("=" * 60)
    
    return results

def export_model():
    """학습된 모델을 최종 경로로 복사"""
    print("\n" + "=" * 60)
    print("모델 내보내기 중...")
    print("=" * 60)
    
    import shutil
    
    # 학습된 모델 경로
    best_model = os.path.join(PROJECT_NAME, "train", "weights", "best.pt")
    last_model = os.path.join(PROJECT_NAME, "train", "weights", "last.pt")
    
    # 출력 경로
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_best = os.path.join(OUTPUT_DIR, "microscope_best.pt")
    output_last = os.path.join(OUTPUT_DIR, "microscope_last.pt")
    
    if os.path.exists(best_model):
        shutil.copy(best_model, output_best)
        print(f"\n✓ Best 모델 저장: {output_best}")
    
    if os.path.exists(last_model):
        shutil.copy(last_model, output_last)
        print(f"✓ Last 모델 저장: {output_last}")
    
    return output_best

def validate_model(model_path):
    """모델 검증"""
    print("\n" + "=" * 60)
    print("모델 검증 중...")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # Validation 데이터셋으로 평가
    results = model.val(
        data=DATA_YAML,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        plots=True,
    )
    
    print("\n검증 결과:")
    print(f"  - mAP50: {results.box.map50:.4f}")
    print(f"  - mAP50-95: {results.box.map:.4f}")
    print(f"  - Precision: {results.box.mp:.4f}")
    print(f"  - Recall: {results.box.mr:.4f}")
    
    return results

def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("현미경 이미지 YOLOv8 모델 학습")
    print("=" * 60 + "\n")
    
    # 데이터셋 확인
    if not check_data():
        return
    
    # 모델 학습
    train_results = train_model()
    
    # 모델 내보내기
    model_path = export_model()
    
    # 모델 검증
    val_results = validate_model(model_path)
    
    print("\n" + "=" * 60)
    print("✓ 모든 작업 완료!")
    print("=" * 60)
    print(f"\n학습된 모델: {model_path}")
    print("\n다음 단계:")
    print("  1. 학습 결과 확인: tensorboard --logdir=microscope_detection/train")
    print("  2. PyQt 앱에서 모델 사용")

if __name__ == "__main__":
    main()
