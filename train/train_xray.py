"""
X-ray 이미지 YOLOv8 모델 학습 스크립트
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "xray"
MODEL_DIR = PROJECT_ROOT / "models"

# 학습 설정
DATA_YAML = str(DATA_DIR / "data.yaml")
MODEL_SIZE = "yolov8n"  # n, s, m, l, x 중 선택
EPOCHS = 150
BATCH_SIZE = 16
IMG_SIZE = 640

# 디바이스 설정 (Apple Silicon MPS 지원)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

PROJECT_NAME = str(PROJECT_ROOT / "runs" / "xray_detection")
OUTPUT_DIR = str(MODEL_DIR)

def check_data():
    """데이터셋 확인"""
    print("=" * 60)
    print("데이터셋 확인 중...")
    print("=" * 60)

    data_yaml_path = Path(DATA_YAML)
    if not data_yaml_path.exists():
        print(f"\n데이터셋을 찾을 수 없습니다: {DATA_YAML}")
        print("먼저 download_xray.py를 실행하세요:")
        print("  python download_xray.py --num_images 500")
        return False

    # 이미지 개수 확인
    train_img_dir = DATA_DIR / "train" / "images"
    if not train_img_dir.exists():
        print(f"\n학습 이미지 폴더를 찾을 수 없습니다: {train_img_dir}")
        return False

    num_images = len(list(train_img_dir.glob("*.png")))
    if num_images == 0:
        print(f"\n학습 이미지가 없습니다: {train_img_dir}")
        print("download_xray.py를 실행하여 데이터를 생성하세요.")
        return False

    print(f"\n데이터셋 발견: {DATA_YAML}")
    print(f"학습 이미지: {num_images}장")
    print(f"디바이스: {DEVICE}")
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
        patience=30,  # Early stopping (X-ray는 더 오래 학습)
        save=True,
        save_period=10,
        
        # 데이터 증강 (X-ray 특성에 맞게 조정)
        augment=True,
        hsv_h=0.01,   # X-ray는 색상 변화 적음
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5.0,   # 작은 회전
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,    # X-ray는 상하 반전 가능
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,     # X-ray에 mixup 적용
        
        # 최적화
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,  # X-ray는 warmup 더 길게
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

    # 학습된 모델 경로 (최신 학습 결과 찾기)
    runs_dir = Path(PROJECT_NAME)
    if not runs_dir.exists():
        print(f"학습 결과를 찾을 수 없습니다: {runs_dir}")
        return None

    # 가장 최근 train 폴더 찾기
    train_dirs = sorted(runs_dir.glob("train*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not train_dirs:
        print("학습 결과 폴더가 없습니다.")
        return None

    latest_train = train_dirs[0]
    best_model = latest_train / "weights" / "best.pt"
    last_model = latest_train / "weights" / "last.pt"

    # 출력 경로
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_best = MODEL_DIR / "xray_best.pt"
    output_last = MODEL_DIR / "xray_last.pt"

    if best_model.exists():
        shutil.copy(best_model, output_best)
        print(f"\nBest 모델 저장: {output_best}")
    else:
        print(f"Best 모델을 찾을 수 없습니다: {best_model}")

    if last_model.exists():
        shutil.copy(last_model, output_last)
        print(f"Last 모델 저장: {output_last}")

    return str(output_best) if output_best.exists() else None

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
    
    # 클래스별 성능
    print("\n클래스별 성능:")
    class_names = ["Void", "Bridge", "HiP", "ColdJoint", "Crack", "Normal"]
    for i, name in enumerate(class_names):
        if i < len(results.box.ap_class_index):
            print(f"  - {name}: mAP50={results.box.ap50[i]:.4f}")
    
    return results

def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("X-ray 이미지 YOLOv8 모델 학습")
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
    print("  1. 학습 결과 확인: tensorboard --logdir=xray_detection/train")
    print("  2. PyQt 앱에서 모델 사용")

if __name__ == "__main__":
    main()
