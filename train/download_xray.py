"""
X-ray 이미지 데이터셋 생성 스크립트
물리 기반 시뮬레이션을 사용하여 YOLOv8 형식 X-ray 데이터셋을 생성합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "lib"))

# 데이터 저장 경로
DATA_DIR = PROJECT_ROOT / "data" / "xray"


def create_dataset_structure():
    """YOLOv8 형식의 데이터셋 구조 생성"""
    print("\n" + "=" * 60)
    print("데이터셋 구조 생성 중...")
    print("=" * 60)

    dirs = [
        DATA_DIR / "train" / "images",
        DATA_DIR / "train" / "labels",
        DATA_DIR / "valid" / "images",
        DATA_DIR / "valid" / "labels",
        DATA_DIR / "test" / "images",
        DATA_DIR / "test" / "labels",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n데이터셋 디렉토리 생성 완료: {DATA_DIR}")


def generate_synthetic_data(num_images: int = 500, grid_size: tuple = (5, 5)):
    """
    물리 기반 시뮬레이션으로 합성 X-ray 데이터 생성

    Args:
        num_images: 생성할 이미지 수
        grid_size: 각 이미지의 범프 배열 크기 (rows, cols)
    """
    print("\n" + "=" * 60)
    print("합성 X-ray 데이터 생성")
    print("=" * 60)

    try:
        from generate_yolo_dataset import YOLODatasetGenerator

        generator = YOLODatasetGenerator(DATA_DIR)
        generator.generate_dataset(
            num_images=num_images,
            grid_size=grid_size,
            train_ratio=0.7,
            valid_ratio=0.2
        )

        return True

    except ImportError as e:
        print(f"\n오류: 데이터 생성기를 불러올 수 없습니다: {e}")
        print("generate_yolo_dataset.py 파일을 확인하세요.")
        return False


def check_dataset():
    """생성된 데이터셋 확인"""
    print("\n" + "=" * 60)
    print("데이터셋 확인")
    print("=" * 60)

    for split in ["train", "valid", "test"]:
        img_dir = DATA_DIR / split / "images"
        label_dir = DATA_DIR / split / "labels"

        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.png")))
            label_count = len(list(label_dir.glob("*.txt")))
            print(f"  {split:6s}: {img_count} images, {label_count} labels")
        else:
            print(f"  {split:6s}: 없음")

    # data.yaml 확인
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"\n  data.yaml: 존재")
    else:
        print(f"\n  data.yaml: 없음")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="X-ray 데이터셋 생성")
    parser.add_argument("--num_images", type=int, default=500,
                       help="생성할 이미지 수 (기본: 500)")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[5, 5],
                       help="범프 배열 크기 (기본: 5 5)")
    parser.add_argument("--check_only", action="store_true",
                       help="데이터셋 확인만 수행")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("X-ray 이미지 데이터셋 준비")
    print("=" * 60)

    if args.check_only:
        check_dataset()
        return

    # 데이터셋 구조 생성
    create_dataset_structure()

    # 합성 데이터 생성
    success = generate_synthetic_data(
        num_images=args.num_images,
        grid_size=tuple(args.grid_size)
    )

    if success:
        check_dataset()

        print("\n" + "=" * 60)
        print("데이터셋 준비 완료!")
        print("=" * 60)
        print(f"\n데이터 경로: {DATA_DIR}")
        print("\n다음 단계:")
        print("  python train_xray.py  # 모델 학습")
    else:
        print("\n데이터 생성에 실패했습니다.")


if __name__ == "__main__":
    main()
