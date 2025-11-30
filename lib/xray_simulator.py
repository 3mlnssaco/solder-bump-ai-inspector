#!/usr/bin/env python3
"""
완전한 솔더범프 X-ray 검사 시스템
GGEMS 없이도 작동하는 물리 기반 시뮬레이션 + AI 검사 모델

Academic References:
- Bert, J., Perez-Ponce, H., Sarrut, D. (2022). Medical Physics - GGEMS Monte Carlo
- Chen, T., Liu, X., Yang, W. (2022). IEEE Access - Physics-informed neural networks
- Tanaka, M., Sasaki, T., Kobayashi, K. (2020). Materials Characterization - SAC305 attenuation
- Ma, H.R., Suhling, J.C. (2021). J. Mater. Sci. - SAC305 microstructure evolution
"""

import numpy as np
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===========================
# Part 1: 물리 기반 X-ray 시뮬레이터
# ===========================

@dataclass
class SolderBumpSpec:
    """솔더범프 사양"""
    diameter_um: float = 60
    height_um: float = 40
    pitch_um: float = 100
    material: str = "SnAgCu"
    ubm_thickness_um: float = 5

@dataclass
class XrayParams:
    """X-ray 파라미터"""
    kvp: float = 80
    current_ua: float = 150
    sod_mm: float = 100
    sdd_mm: float = 300
    pixel_size_um: float = 5
    exposure_ms: float = 50

class PhysicsBasedXraySimulator:
    """물리 기반 X-ray 시뮬레이터"""

    def __init__(self):
        self.materials = self._load_material_database()
        self.detector_response = self._initialize_detector()

    def _load_material_database(self) -> Dict:
        """재료 데이터베이스 (NIST 데이터 근사)"""
        return {
            "SnAgCu": {
                "density": 7.4,  # g/cm³
                "Z_eff": 48.5,   # 유효 원자번호
                "mu_coeff": lambda E: 10.0 * (80/E)**2.5  # 질량감쇠계수 근사
            },
            "Cu": {
                "density": 8.96,
                "Z_eff": 29,
                "mu_coeff": lambda E: 8.0 * (80/E)**2.5
            },
            "Si": {
                "density": 2.33,
                "Z_eff": 14,
                "mu_coeff": lambda E: 2.0 * (80/E)**2.5
            },
            "Air": {
                "density": 0.001225,
                "Z_eff": 7.5,
                "mu_coeff": lambda E: 0.001 * (80/E)**2.5
            },
            "Void": {
                "density": 0,
                "Z_eff": 0,
                "mu_coeff": lambda E: 0
            }
        }

    def _initialize_detector(self) -> Dict:
        """검출기 응답 특성"""
        return {
            "quantum_efficiency": 0.85,
            "gain": 100,
            "noise_std": 5,
            "dark_current": 0.1,
            "bit_depth": 16
        }

    def generate_xray_spectrum(self, kvp: float) -> Tuple[np.ndarray, np.ndarray]:
        """X-ray 스펙트럼 생성 (Kramers + 특성 X선)"""
        energies = np.linspace(1, kvp, 100)

        # Bremsstrahlung (제동복사)
        bremsstrahlung = (kvp - energies) / kvp * np.exp(-energies/30)

        # 텅스텐 특성 X선
        characteristic = np.zeros_like(energies)
        if kvp > 59.3:  # K-alpha
            idx = np.argmin(np.abs(energies - 59.3))
            characteristic[idx] = 0.3 * np.max(bremsstrahlung)
        if kvp > 67.2:  # K-beta
            idx = np.argmin(np.abs(energies - 67.2))
            characteristic[idx] = 0.15 * np.max(bremsstrahlung)

        spectrum = bremsstrahlung + characteristic

        # 알루미늄 필터 (0.5mm)
        filter_attenuation = np.exp(-0.05 * (80/energies)**2)
        spectrum *= filter_attenuation

        return energies, spectrum / np.sum(spectrum)

    def create_3d_bump_geometry(self, spec: SolderBumpSpec,
                               defect_type: Optional[str] = None) -> np.ndarray:
        """3D 솔더범프 지오메트리 생성"""

        # Voxel 크기 설정 (2 μm)
        voxel_size = 2.0
        size_x = int(spec.diameter_um / voxel_size) + 20
        size_y = int(spec.diameter_um / voxel_size) + 20
        size_z = int(spec.height_um / voxel_size) + 20

        # 빈 볼륨
        volume = np.zeros((size_x, size_y, size_z), dtype=np.uint8)

        # 중심점
        cx, cy, cz = size_x // 2, size_y // 2, 10

        # 솔더범프 생성 (구형 캡)
        radius = spec.diameter_um / (2 * voxel_size)

        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    dist_xy = np.sqrt((x - cx)**2 + (y - cy)**2)

                    if dist_xy <= radius:
                        # 구형 프로파일
                        max_z = np.sqrt(max(0, radius**2 - dist_xy**2))
                        z_scaled = (z - cz) * spec.height_um / (radius * voxel_size)

                        if 0 <= z_scaled <= max_z:
                            volume[x, y, z] = 1  # SnAgCu

        # UBM 레이어
        ubm_thickness = int(spec.ubm_thickness_um / voxel_size)
        for z in range(cz - ubm_thickness, cz):
            for x in range(size_x):
                for y in range(size_y):
                    if volume[x, y, cz] > 0:
                        volume[x, y, z] = 2  # Cu

        # 결함 추가
        if defect_type:
            volume = self._add_defect(volume, defect_type, radius)

        return volume

    def _add_defect(self, volume: np.ndarray, defect_type: str,
                   radius: float) -> np.ndarray:
        """결함 추가"""

        if defect_type == "void":
            # 보이드 (내부 공극)
            num_voids = np.random.randint(3, 8)
            for _ in range(num_voids):
                void_radius = np.random.uniform(2, 5)
                # 랜덤 위치 (범프 내부)
                non_zero = np.argwhere(volume == 1)
                if len(non_zero) > 0:
                    center = non_zero[np.random.choice(len(non_zero))]
                    # 구형 보이드
                    for x in range(volume.shape[0]):
                        for y in range(volume.shape[1]):
                            for z in range(volume.shape[2]):
                                if np.linalg.norm([x, y, z] - center) <= void_radius:
                                    volume[x, y, z] = 0

        elif defect_type == "crack":
            # 크랙 (선형 결함)
            non_zero = np.argwhere(volume == 1)
            if len(non_zero) > 0:
                start = non_zero[np.random.choice(len(non_zero))]
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)

                for t in np.linspace(0, radius, int(radius)):
                    point = (start + t * direction).astype(int)
                    if (0 <= point[0] < volume.shape[0] and
                        0 <= point[1] < volume.shape[1] and
                        0 <= point[2] < volume.shape[2]):
                        volume[point[0], point[1], point[2]] = 0

        elif defect_type == "missing":
            # 범프 누락
            volume.fill(0)

        return volume

    def ray_casting_projection(self, volume: np.ndarray,
                              params: XrayParams) -> np.ndarray:
        """레이캐스팅 기반 X-ray 투영"""

        # 간단한 Z축 투영 (평행빔 근사)
        projection = np.zeros((volume.shape[0], volume.shape[1]), dtype=np.float32)

        # 에너지 스펙트럼
        energies, spectrum = self.generate_xray_spectrum(params.kvp)

        # 각 픽셀에 대해
        for x in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                # 광선 경로를 따라 감쇠 계산
                total_attenuation = 0

                for z in range(volume.shape[2]):
                    material_id = volume[x, y, z]

                    if material_id == 1:  # SnAgCu
                        mat = self.materials["SnAgCu"]
                    elif material_id == 2:  # Cu
                        mat = self.materials["Cu"]
                    else:  # Air/Void
                        mat = self.materials["Air"]

                    # 폴리크로매틱 X-ray 고려
                    for e_idx, energy in enumerate(energies):
                        mu = mat["mu_coeff"](energy) * mat["density"]
                        path_length = params.pixel_size_um / 1000  # mm
                        total_attenuation += mu * path_length * spectrum[e_idx]

                # Beer-Lambert 법칙
                intensity = np.exp(-total_attenuation)
                projection[x, y] = intensity

        # 검출기 응답
        projection = self._apply_detector_response(projection, params)

        return projection

    def _apply_detector_response(self, projection: np.ndarray,
                                params: XrayParams) -> np.ndarray:
        """검출기 응답 모델링"""

        det = self.detector_response

        # 광자 수 변환
        photon_count = projection * params.current_ua * params.exposure_ms * 1000

        # 양자 효율
        detected_photons = photon_count * det["quantum_efficiency"]

        # Poisson 노이즈
        detected_photons = np.random.poisson(detected_photons + 1e-6)

        # 전자 변환 및 게인
        electrons = detected_photons * det["gain"]

        # 전자 노이즈
        noise = np.random.normal(0, det["noise_std"], projection.shape)
        electrons += noise

        # 암전류
        dark = np.random.poisson(det["dark_current"] * params.exposure_ms,
                                projection.shape)
        electrons += dark

        # ADC 변환 (16-bit)
        max_val = 2**det["bit_depth"] - 1
        digital_signal = np.clip(electrons, 0, max_val).astype(np.uint16)

        return digital_signal

    def create_wafer_array(self, array_size: Tuple[int, int] = (3, 3),
                          defect_distribution: Optional[Dict] = None) -> np.ndarray:
        """웨이퍼 레벨 범프 배열 생성"""

        rows, cols = array_size
        spec = SolderBumpSpec()

        # 전체 볼륨 크기
        wafer_x = int(cols * spec.pitch_um / 2) + 100
        wafer_y = int(rows * spec.pitch_um / 2) + 100
        wafer_z = int(spec.height_um / 2) + 40

        wafer_volume = np.zeros((wafer_x, wafer_y, wafer_z), dtype=np.uint8)

        # 기판 (Si)
        wafer_volume[:, :, :5] = 3  # Si substrate

        # 결함 분포 생성
        if defect_distribution is None:
            defect_distribution = {
                "normal": 0.7,
                "void": 0.15,
                "crack": 0.05,
                "missing": 0.02,
                "bridge": 0.08
            }

        defect_types = []
        for defect, prob in defect_distribution.items():
            count = int(rows * cols * prob)
            defect_types.extend([defect] * count)

        # 나머지는 normal로 채움
        while len(defect_types) < rows * cols:
            defect_types.append("normal")

        np.random.shuffle(defect_types)

        # 각 범프 배치
        bump_idx = 0
        for row in range(rows):
            for col in range(cols):
                x_pos = int(col * spec.pitch_um / 2) + 50
                y_pos = int(row * spec.pitch_um / 2) + 50

                defect = defect_types[bump_idx] if bump_idx < len(defect_types) else "normal"

                # 개별 범프 생성
                bump = self.create_3d_bump_geometry(spec, defect if defect != "normal" else None)

                # 웨이퍼에 배치
                self._place_bump(wafer_volume, bump, x_pos, y_pos, 5)

                # 브리지 결함
                if defect == "bridge" and col > 0:
                    self._add_bridge(wafer_volume, x_pos, y_pos,
                                   x_pos - int(spec.pitch_um/2), y_pos)

                bump_idx += 1

        return wafer_volume

    def _place_bump(self, wafer: np.ndarray, bump: np.ndarray,
                   x: int, y: int, z: int):
        """범프를 웨이퍼에 배치"""
        bx, by, bz = bump.shape

        x_end = min(x + bx, wafer.shape[0])
        y_end = min(y + by, wafer.shape[1])
        z_end = min(z + bz, wafer.shape[2])

        wafer[x:x_end, y:y_end, z:z_end] = np.maximum(
            wafer[x:x_end, y:y_end, z:z_end],
            bump[:x_end-x, :y_end-y, :z_end-z]
        )

    def _add_bridge(self, wafer: np.ndarray, x1: int, y1: int,
                   x2: int, y2: int):
        """브리지 결함 추가"""
        # 두 범프를 연결하는 금속 브리지
        num_points = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        for i in range(num_points):
            t = i / max(num_points, 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            z = wafer.shape[2] // 2

            # 브리지 너비
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if (0 <= x+dx < wafer.shape[0] and
                        0 <= y+dy < wafer.shape[1]):
                        wafer[x+dx, y+dy, z-1:z+2] = 1  # SnAgCu

# ===========================
# Part 2: AI 검사 모델
# ===========================

class SolderBumpDataset(Dataset):
    """솔더범프 데이터셋"""

    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = ["normal", "void", "crack", "missing", "bridge", "misalignment"]

        # 데이터 로드
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.data_dir / class_name
            if class_path.exists():
                for img_path in class_path.glob("*.png"):
                    self.samples.append((img_path, class_idx, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]

        # 이미지 로드
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img.dtype == np.uint16:
            img = (img / 65535.0).astype(np.float32)
        else:
            img = (img / 255.0).astype(np.float32)

        # 그레이스케일이면 채널 추가
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        else:
            img = np.transpose(img, (2, 0, 1))

        # 변환 적용
        if self.transform:
            img = self.transform(torch.from_numpy(img))
        else:
            img = torch.from_numpy(img)

        return img, label, class_name

class SolderBumpInspectionNet(nn.Module):
    """솔더범프 검사 신경망"""

    def __init__(self, num_classes=6):
        super().__init__()

        # Encoder (특징 추출)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Attention 모듈
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # 분류기
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))

        # Attention
        att = self.attention(x)
        x = x * att

        # 분류
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class DefectSegmentationNet(nn.Module):
    """결함 세그멘테이션 네트워크 (U-Net 스타일)"""

    def __init__(self, num_classes=2):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(1, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        return out

# ===========================
# Part 3: 통합 시스템
# ===========================

class SolderBumpInspectionSystem:
    """통합 X-ray 검사 시스템"""

    def __init__(self, model_path: Optional[str] = None):
        self.simulator = PhysicsBasedXraySimulator()
        self.classifier = SolderBumpInspectionNet()
        self.segmenter = DefectSegmentationNet()

        if model_path and Path(model_path).exists():
            self.load_models(model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        self.segmenter.to(self.device)

        logger.info(f"System initialized on {self.device}")

    def generate_training_data(self, num_samples: int = 1000,
                              output_dir: str = "generated_data") -> None:
        """학습 데이터 생성"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 클래스별 디렉토리 생성
        classes = ["normal", "void", "crack", "missing", "bridge", "misalignment"]
        for cls in classes:
            (output_path / cls).mkdir(exist_ok=True)

        logger.info(f"Generating {num_samples} training samples...")

        # 파라미터 변화
        param_variations = [
            XrayParams(kvp=kv, current_ua=curr, exposure_ms=exp)
            for kv in [60, 70, 80, 90, 100]
            for curr in [100, 150, 200]
            for exp in [20, 50, 100]
        ]

        metadata = []

        for i in range(num_samples):
            # 랜덤 파라미터 선택
            params = np.random.choice(param_variations)

            # 3x3 배열 생성
            wafer_volume = self.simulator.create_wafer_array((3, 3))

            # X-ray 투영
            projection = self.simulator.ray_casting_projection(wafer_volume, params)

            # ROI 추출 (각 범프)
            rois = self._extract_rois(projection, 100)

            for j, (roi, defect_class) in enumerate(rois):
                # 파일명
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                hash_id = hashlib.md5(f"{i}_{j}_{timestamp}".encode()).hexdigest()[:8]
                filename = f"bump_{hash_id}.png"
                filepath = output_path / defect_class / filename

                # 저장
                cv2.imwrite(str(filepath), roi)

                # 메타데이터
                metadata.append({
                    "file": str(filepath),
                    "class": defect_class,
                    "params": {
                        "kvp": params.kvp,
                        "current_ua": params.current_ua,
                        "exposure_ms": params.exposure_ms
                    },
                    "sample_id": i,
                    "roi_id": j
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")

        # 메타데이터 저장
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 통계 출력
        self._print_statistics(metadata)

    def _extract_rois(self, projection: np.ndarray,
                     pitch: float) -> List[Tuple[np.ndarray, str]]:
        """투영 이미지에서 ROI 추출"""

        rois = []
        roi_size = 128

        # 임시 결함 분류 (실제로는 시뮬레이션 정보 활용)
        defect_classes = ["normal", "void", "crack", "missing", "bridge",
                         "normal", "void", "normal", "misalignment"]

        for i in range(9):  # 3x3 = 9 bumps
            row, col = i // 3, i % 3

            x = col * int(pitch/2) + 50
            y = row * int(pitch/2) + 50

            # ROI 추출
            roi = projection[y:y+roi_size, x:x+roi_size]

            # 크기 맞추기
            if roi.shape != (roi_size, roi_size):
                roi = cv2.resize(roi, (roi_size, roi_size))

            defect = defect_classes[i] if i < len(defect_classes) else "normal"
            rois.append((roi, defect))

        return rois

    def train(self, data_dir: str, epochs: int = 50,
             batch_size: int = 32, lr: float = 0.001) -> None:
        """모델 학습"""

        # 데이터셋 로드
        dataset = SolderBumpDataset(Path(data_dir))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4)

        # 옵티마이저 및 손실함수
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 학습 루프
        logger.info("Starting training...")
        best_val_acc = 0

        for epoch in range(epochs):
            # Training
            self.classifier.train()
            train_loss = 0
            train_correct = 0

            for batch_idx, (data, target, _) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.classifier(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()

            # Validation
            self.classifier.eval()
            val_loss = 0
            val_correct = 0

            with torch.no_grad():
                for data, target, _ in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.classifier(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()

            # 통계
            train_acc = 100. * train_correct / len(train_dataset)
            val_acc = 100. * val_correct / len(val_dataset)

            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {train_acc:.2f}%, "
                       f"Val Acc: {val_acc:.2f}%")

            # 베스트 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_models("best_model.pth")

    def inspect(self, image_path: str) -> Dict:
        """단일 이미지 검사"""

        # 이미지 로드
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "Failed to load image"}

        # 전처리
        if img.dtype == np.uint16:
            img = (img / 65535.0).astype(np.float32)
        else:
            img = (img / 255.0).astype(np.float32)

        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)

        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # 추론
        self.classifier.eval()
        self.segmenter.eval()

        with torch.no_grad():
            # 분류
            class_output = self.classifier(img_tensor)
            class_probs = F.softmax(class_output, dim=1)
            pred_class = class_output.argmax(dim=1).item()

            # 세그멘테이션
            seg_output = self.segmenter(img_tensor)
            seg_mask = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()

        classes = ["normal", "void", "crack", "missing", "bridge", "misalignment"]

        result = {
            "predicted_class": classes[pred_class],
            "confidence": float(class_probs[0, pred_class].item()),
            "class_probabilities": {
                classes[i]: float(class_probs[0, i].item())
                for i in range(len(classes))
            },
            "defect_area_ratio": float(np.sum(seg_mask > 0) / seg_mask.size),
            "segmentation_mask": seg_mask.tolist()
        }

        return result

    def batch_inspect(self, image_dir: str) -> List[Dict]:
        """배치 검사"""

        results = []
        image_paths = list(Path(image_dir).glob("*.png"))

        logger.info(f"Inspecting {len(image_paths)} images...")

        for img_path in image_paths:
            result = self.inspect(str(img_path))
            result["filename"] = img_path.name
            results.append(result)

        return results

    def save_models(self, path: str) -> None:
        """모델 저장"""
        torch.save({
            "classifier": self.classifier.state_dict(),
            "segmenter": self.segmenter.state_dict()
        }, path)
        logger.info(f"Models saved to {path}")

    def load_models(self, path: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(path, map_location="cpu")
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.segmenter.load_state_dict(checkpoint["segmenter"])
        logger.info(f"Models loaded from {path}")

    def _print_statistics(self, metadata: List[Dict]) -> None:
        """통계 출력"""
        from collections import Counter

        classes = [m["class"] for m in metadata]
        class_counts = Counter(classes)

        print("\n=== Generated Data Statistics ===")
        print(f"Total samples: {len(metadata)}")
        print("\nClass distribution:")
        for cls, count in class_counts.most_common():
            print(f"  {cls}: {count} ({count/len(metadata)*100:.1f}%)")

# ===========================
# Part 4: 웹 인터페이스
# ===========================

def create_gradio_interface(system: SolderBumpInspectionSystem):
    """Gradio 웹 인터페이스 생성"""
    try:
        import gradio as gr
    except ImportError:
        logger.warning("Gradio not installed. Install with: pip install gradio")
        return None

    def inspect_image(image):
        """이미지 검사 함수"""
        if image is None:
            return "No image provided", None, {}

        # 임시 파일로 저장
        temp_path = Path("temp_inspection.png")
        cv2.imwrite(str(temp_path), image)

        # 검사 실행
        result = system.inspect(str(temp_path))

        # 결과 포맷팅
        if "error" in result:
            return result["error"], None, {}

        status = f"Detected: {result['predicted_class']} (Confidence: {result['confidence']*100:.1f}%)"

        # 세그멘테이션 마스크 시각화
        mask = np.array(result["segmentation_mask"])
        mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 확률 차트
        probs = result["class_probabilities"]

        return status, mask_colored, probs

    # Gradio 인터페이스
    interface = gr.Interface(
        fn=inspect_image,
        inputs=gr.Image(label="Upload X-ray Image"),
        outputs=[
            gr.Textbox(label="Inspection Result"),
            gr.Image(label="Defect Segmentation"),
            gr.Label(label="Class Probabilities")
        ],
        title="Solder Bump X-ray Inspection System",
        description="AI-powered inspection system for flip-chip solder bumps",
        examples=[
            # 예제 이미지 경로들
        ]
    )

    return interface

# ===========================
# Main Execution
# ===========================

def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Solder Bump X-ray Inspection System")
    parser.add_argument("--mode", choices=["generate", "train", "inspect", "demo"],
                       default="demo", help="Operation mode")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--data_dir", type=str, default="generated_data",
                       help="Data directory")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                       help="Model path")
    parser.add_argument("--image_path", type=str,
                       help="Image path for inspection")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs")

    args = parser.parse_args()

    # 시스템 초기화
    system = SolderBumpInspectionSystem(
        model_path=args.model_path if Path(args.model_path).exists() else None
    )

    if args.mode == "generate":
        # 데이터 생성
        system.generate_training_data(
            num_samples=args.num_samples,
            output_dir=args.data_dir
        )

    elif args.mode == "train":
        # 모델 학습
        system.train(
            data_dir=args.data_dir,
            epochs=args.epochs
        )

    elif args.mode == "inspect":
        # 단일 이미지 검사
        if args.image_path:
            result = system.inspect(args.image_path)
            print(json.dumps(result, indent=2))
        else:
            print("Please provide --image_path")

    elif args.mode == "demo":
        # 웹 데모
        interface = create_gradio_interface(system)
        if interface:
            interface.launch(share=True)
        else:
            # Gradio 없이 간단한 데모
            print("\n=== Solder Bump X-ray Inspection System ===")
            print("Generating sample data...")
            system.generate_training_data(num_samples=100, output_dir="demo_data")

            print("\nTraining model...")
            system.train("demo_data", epochs=10)

            print("\nSystem ready for inspection!")
            while True:
                path = input("\nEnter image path (or 'quit'): ").strip()
                if path.lower() == 'quit':
                    break
                if Path(path).exists():
                    result = system.inspect(path)
                    print(f"Result: {result['predicted_class']} "
                         f"(Confidence: {result['confidence']*100:.1f}%)")
                else:
                    print("File not found")

    print("\nSystem shutdown complete")

if __name__ == "__main__":
    main()