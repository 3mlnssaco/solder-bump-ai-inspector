#!/usr/bin/env python3
"""
YOLOv8 형식 X-ray 솔더범프 데이터셋 생성기

물리 기반 시뮬레이션을 사용하여 YOLOv8 Object Detection용 데이터 생성
- 웨이퍼 이미지 생성
- 각 범프의 bbox + 결함 클래스를 YOLO 형식으로 저장

Classes:
    0: Void (내부 기공)
    1: Bridge (인접 볼 간 단락)
    2: HiP (Head-in-Pillow)
    3: ColdJoint (냉접합)
    4: Crack (균열)
    5: Normal (정상)
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import yaml
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# lib 경로 추가
sys.path.append(str(Path(__file__).parent.parent / "lib"))

# 데이터 저장 경로
DATA_DIR = Path(__file__).parent.parent / "data" / "xray"

# 클래스 정의
CLASS_NAMES = {
    0: "Void",
    1: "Bridge",
    2: "HiP",
    3: "ColdJoint",
    4: "Crack",
    5: "Normal"
}

# 결함 분포 (실제 생산 라인 기반)
DEFECT_DISTRIBUTION = {
    "Normal": 0.70,     # 70% 정상
    "Void": 0.12,       # 12% 보이드
    "Bridge": 0.06,     # 6% 브리지
    "HiP": 0.04,        # 4% HiP
    "ColdJoint": 0.04,  # 4% 냉접합
    "Crack": 0.04       # 4% 크랙
}

# ============================================================
# 프로젝트 대표 스펙 (6-inch 웨이퍼 기준)
# ============================================================
# 웨이퍼: 6-inch (직경 152mm)
# 다이 크기: 5×5 mm
# 범프 피치: 150~300 µm
# 범프 직경: 90~100 µm (피치의 60-70%)
# 범프 높이: 60~80 µm (CPB 기준)
# 다이당 범프: 14×14 = 196개
# 웨이퍼당 다이: ~700개
# 웨이퍼당 총 범프: ~1.3~1.5×10⁵개
# ============================================================

@dataclass
class BumpSpec:
    """솔더범프 사양 (프로젝트 기준)"""
    diameter_um: float = 100      # 범프 직경 (µm)
    height_um: float = 70         # 범프 높이 (µm)
    pitch_um: float = 150         # 범프 피치 (µm)
    die_size_mm: float = 5.0      # 다이 크기 (mm)
    bumps_per_row: int = 14       # 다이당 범프 배열 (14×14)

# 검사 단위: Die (14×14 = 196개 범프)
DEFAULT_GRID_SIZE = (14, 14)
DEFAULT_BUMP_SIZE = 40           # 픽셀 (시뮬레이션용)
DEFAULT_PADDING = 6              # 픽셀 (시뮬레이션용)


@dataclass
class XrayParams:
    """X-ray 파라미터"""
    kvp: float = 80
    current_ua: float = 150
    pixel_size_um: float = 5
    exposure_ms: float = 50


class SimplifiedXraySimulator:
    """
    간소화된 X-ray 시뮬레이터
    Beer-Lambert 법칙 기반
    """

    def __init__(self):
        self.materials = {
            "SnAgCu": {"density": 7.4, "mu": 10.0},
            "Cu": {"density": 8.96, "mu": 8.0},
            "Void": {"density": 0, "mu": 0},
            "Air": {"density": 0.001, "mu": 0.001}
        }

    def generate_bump_image(self, size: int = 64, defect_type: str = "Normal") -> np.ndarray:
        """
        단일 범프 X-ray 이미지 생성

        Args:
            size: 이미지 크기 (정사각형)
            defect_type: 결함 유형

        Returns:
            np.ndarray: 8-bit 그레이스케일 이미지
        """
        img = np.ones((size, size), dtype=np.float32)
        center = size // 2
        radius = size // 3

        # 기본 범프 (구형 감쇠)
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)

        # Beer-Lambert 감쇠 (범프 내부)
        bump_mask = dist < radius
        thickness = np.sqrt(np.maximum(0, radius**2 - dist**2)) * 2
        attenuation = np.exp(-0.1 * thickness)

        img[bump_mask] = attenuation[bump_mask]

        # 결함 추가
        if defect_type == "Void":
            img = self._add_void(img, center, radius)
        elif defect_type == "Bridge":
            img = self._add_bridge(img, center, radius, size)
        elif defect_type == "HiP":
            img = self._add_hip(img, center, radius)
        elif defect_type == "ColdJoint":
            img = self._add_cold_joint(img, center, radius)
        elif defect_type == "Crack":
            img = self._add_crack(img, center, radius)

        # 노이즈 추가
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)

        # 8-bit 변환
        img_8bit = (img * 255).astype(np.uint8)

        return img_8bit

    def _add_void(self, img: np.ndarray, center: int, radius: int) -> np.ndarray:
        """보이드 추가 (어두운 구형 영역)"""
        num_voids = random.randint(1, 4)
        for _ in range(num_voids):
            void_radius = random.randint(3, 8)
            offset_x = random.randint(-radius//2, radius//2)
            offset_y = random.randint(-radius//2, radius//2)
            void_x = center + offset_x
            void_y = center + offset_y

            y, x = np.ogrid[:img.shape[0], :img.shape[1]]
            void_mask = (x - void_x)**2 + (y - void_y)**2 < void_radius**2
            img[void_mask] = np.minimum(img[void_mask] + 0.3, 1.0)

        return img

    def _add_bridge(self, img: np.ndarray, center: int, radius: int, size: int) -> np.ndarray:
        """브리지 추가 (옆으로 연결)"""
        # 오른쪽으로 브리지 연결
        bridge_width = random.randint(3, 8)
        bridge_y_start = center - bridge_width // 2
        bridge_y_end = center + bridge_width // 2

        for y in range(bridge_y_start, bridge_y_end):
            for x in range(center + radius - 5, size - 5):
                if 0 <= y < size and 0 <= x < size:
                    img[y, x] = random.uniform(0.3, 0.5)

        return img

    def _add_hip(self, img: np.ndarray, center: int, radius: int) -> np.ndarray:
        """Head-in-Pillow 추가 (부분 접촉)"""
        # 아래쪽 절반이 밝게 (접촉 불량)
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        hip_mask = (dist < radius) & (y > center)
        img[hip_mask] = np.minimum(img[hip_mask] + 0.25, 1.0)

        return img

    def _add_cold_joint(self, img: np.ndarray, center: int, radius: int) -> np.ndarray:
        """냉접합 추가 (불규칙한 표면)"""
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        bump_mask = dist < radius

        # 불규칙한 패턴 추가
        noise = np.random.normal(0, 0.15, img.shape)
        img[bump_mask] += noise[bump_mask]
        img = np.clip(img, 0, 1)

        return img

    def _add_crack(self, img: np.ndarray, center: int, radius: int) -> np.ndarray:
        """크랙 추가 (선형 밝은 영역)"""
        # 랜덤 각도의 크랙
        angle = random.uniform(0, np.pi)
        length = radius

        for t in np.linspace(-length, length, 50):
            x = int(center + t * np.cos(angle))
            y = int(center + t * np.sin(angle))

            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                # 크랙 주변 2픽셀
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
                            img[ny, nx] = min(img[ny, nx] + 0.3, 1.0)

        return img

    def generate_wafer_image(self, grid_size: Tuple[int, int] = (5, 5),
                            bump_size: int = 64, padding: int = 10,
                            circular: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        웨이퍼 레벨 이미지 생성 (원형 웨이퍼 지원)

        Args:
            grid_size: (rows, cols) 범프 배열 크기
            bump_size: 각 범프 이미지 크기
            padding: 범프 간 간격
            circular: True면 원형 웨이퍼, False면 직사각형

        Returns:
            img: 전체 웨이퍼 이미지
            annotations: 각 범프의 bbox 및 클래스 정보
        """
        rows, cols = grid_size
        img_width = cols * bump_size + (cols + 1) * padding
        img_height = rows * bump_size + (rows + 1) * padding

        # 원형 웨이퍼: 정사각형으로 만들기
        if circular:
            img_size = max(img_width, img_height)
            img_width = img_height = img_size

        # 배경 (웨이퍼 기판 색상)
        wafer_img = np.ones((img_height, img_width), dtype=np.uint8) * 180

        # 원형 웨이퍼 마스크 생성
        center_x, center_y = img_width // 2, img_height // 2
        wafer_radius = min(center_x, center_y) - padding

        if circular:
            # 원형 웨이퍼 배경
            y_grid, x_grid = np.ogrid[:img_height, :img_width]
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            wafer_mask = dist_from_center <= wafer_radius
            wafer_img[~wafer_mask] = 50  # 웨이퍼 외부는 어둡게

        annotations = []

        # 범프 위치 계산 (원형 내부만)
        bump_positions = []
        for row in range(rows):
            for col in range(cols):
                x = padding + col * (bump_size + padding)
                y = padding + row * (bump_size + padding)

                # 범프 중심 좌표
                bump_cx = x + bump_size // 2
                bump_cy = y + bump_size // 2

                # 원형 웨이퍼: 원 내부에 있는 범프만 포함
                if circular:
                    dist = np.sqrt((bump_cx - center_x)**2 + (bump_cy - center_y)**2)
                    if dist + bump_size // 2 > wafer_radius:
                        continue  # 원 밖이면 스킵

                bump_positions.append((x, y, bump_cx, bump_cy))

        # 결함 분포에 따라 클래스 할당
        total_bumps = len(bump_positions)
        if total_bumps == 0:
            return wafer_img, annotations

        defect_assignments = self._assign_defects(total_bumps)

        for bump_idx, (x, y, bump_cx, bump_cy) in enumerate(bump_positions):
            defect_type = defect_assignments[bump_idx]
            bump_img = self.generate_bump_image(bump_size, defect_type)

            # 범프 배치
            wafer_img[y:y+bump_size, x:x+bump_size] = bump_img

            # YOLO annotation 생성 (정규화된 좌표)
            class_id = list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(defect_type)]
            x_center = (x + bump_size / 2) / img_width
            y_center = (y + bump_size / 2) / img_height
            width = bump_size / img_width
            height = bump_size / img_height

            annotations.append({
                "class_id": class_id,
                "class_name": defect_type,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "bbox_pixel": (x, y, bump_size, bump_size)
            })

        return wafer_img, annotations

    def _assign_defects(self, num_bumps: int) -> List[str]:
        """결함 분포에 따라 클래스 할당"""
        defects = []
        for defect_type, ratio in DEFECT_DISTRIBUTION.items():
            count = int(num_bumps * ratio)
            defects.extend([defect_type] * count)

        # 나머지는 Normal로
        while len(defects) < num_bumps:
            defects.append("Normal")

        random.shuffle(defects)
        return defects[:num_bumps]


class YOLODatasetGenerator:
    """YOLOv8 형식 데이터셋 생성기"""

    def __init__(self, output_dir: Path = DATA_DIR):
        self.output_dir = Path(output_dir)
        self.simulator = SimplifiedXraySimulator()

    def create_directories(self):
        """디렉토리 구조 생성"""
        for split in ["train", "valid", "test"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        print(f"Created directory structure at {self.output_dir}")

    def generate_dataset(self, num_images: int = 1000,
                        grid_size: Tuple[int, int] = DEFAULT_GRID_SIZE,
                        bump_size: int = DEFAULT_BUMP_SIZE,
                        padding: int = DEFAULT_PADDING,
                        circular: bool = False,  # Die 단위는 직사각형
                        train_ratio: float = 0.7,
                        valid_ratio: float = 0.2):
        """
        전체 데이터셋 생성

        Args:
            num_images: 생성할 이미지 수
            grid_size: 각 이미지의 범프 배열 크기
            bump_size: 범프 크기 (픽셀)
            padding: 범프 간 간격 (픽셀)
            circular: 원형 웨이퍼 여부
            train_ratio: 학습 데이터 비율
            valid_ratio: 검증 데이터 비율
        """
        self.create_directories()

        # 분할 계산
        train_count = int(num_images * train_ratio)
        valid_count = int(num_images * valid_ratio)
        test_count = num_images - train_count - valid_count

        splits = (
            [("train", train_count)] +
            [("valid", valid_count)] +
            [("test", test_count)]
        )

        # 샘플로 범프 개수 확인
        sample_img, sample_anns = self.simulator.generate_wafer_image(
            grid_size, bump_size, padding, circular)
        bumps_per_image = len(sample_anns)

        print("=" * 60)
        print("YOLOv8 X-ray 데이터셋 생성 (Advanced Fine Pitch)")
        print("=" * 60)
        print(f"웨이퍼 형태: {'원형' if circular else '직사각형'}")
        print(f"범프 크기: {bump_size}px, 간격: {padding}px")
        print(f"총 이미지: {num_images}")
        print(f"  - Train: {train_count}")
        print(f"  - Valid: {valid_count}")
        print(f"  - Test: {test_count}")
        print(f"범프/이미지: ~{bumps_per_image}개 (원형 내부)")
        print(f"총 범프 예상: ~{num_images * bumps_per_image:,}개")
        print("=" * 60)

        stats = {name: 0 for name in CLASS_NAMES.values()}

        for split_name, count in splits:
            print(f"\n[{split_name}] {count}개 이미지 생성 중...")

            for i in tqdm(range(count), desc=f"  {split_name}"):
                # 이미지 생성
                img, annotations = self.simulator.generate_wafer_image(
                    grid_size, bump_size, padding, circular)

                # 파일명
                filename = f"{split_name}_{i:06d}"

                # 이미지 저장
                img_path = self.output_dir / split_name / "images" / f"{filename}.png"
                cv2.imwrite(str(img_path), img)

                # 라벨 저장 (YOLO 형식)
                label_path = self.output_dir / split_name / "labels" / f"{filename}.txt"
                with open(label_path, "w") as f:
                    for ann in annotations:
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                        f.write(line)
                        stats[ann['class_name']] += 1

        # data.yaml 생성
        self._create_data_yaml()

        # 통계 출력
        print("\n" + "=" * 60)
        print("데이터셋 생성 완료!")
        print("=" * 60)
        print(f"\n출력 경로: {self.output_dir}")
        print("\n클래스별 분포:")
        total = sum(stats.values())
        for name, count in stats.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {name:12s}: {count:6d} ({pct:5.1f}%)")
        print(f"  {'Total':12s}: {total:6d}")

    def _create_data_yaml(self):
        """data.yaml 파일 생성"""
        data_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": CLASS_NAMES,
            "nc": len(CLASS_NAMES)
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"\ndata.yaml 생성: {yaml_path}")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 X-ray 솔더범프 데이터셋 생성")
    parser.add_argument("--num_images", type=int, default=500,
                       help="생성할 이미지 수 (기본: 500)")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[5, 5],
                       help="범프 배열 크기 (기본: 5 5)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="출력 디렉토리")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR

    generator = YOLODatasetGenerator(output_dir)
    generator.generate_dataset(
        num_images=args.num_images,
        grid_size=tuple(args.grid_size)
    )

    print("\n다음 단계:")
    print("  1. python train_xray.py  # 모델 학습")


if __name__ == "__main__":
    main()
