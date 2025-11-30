#!/usr/bin/env python3
"""
Physics-Based X-ray Solder Bump Simulator for Research
물리 기반 X-ray 솔더범프 시뮬레이터 (연구 논문용)

Physical Models:
1. Beer-Lambert Law: I = I₀ × exp(-μρt)
2. X-ray Spectrum: Kramers' law (Bremsstrahlung) + Characteristic X-rays
3. Noise Model: Poisson (shot noise) + Gaussian (readout noise)
4. Detector Response: QE, Gain, Dark current

Academic References:
- Hubbell, J.H., Seltzer, S.M. (1995). NIST XCOM - Photon Cross Sections Database
- Kramers, H.A. (1923). Phil. Mag. 46, 836 - X-ray continuum theory
- Liu, S., Chen, L., Zhang, X. (2023). IEEE Trans. CPMT - X-ray inspection
- IPC-7095D (2023) - Design and Assembly Process Implementation for BGAs

Author: Research Project
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml


# ============================================================
# 프로젝트 스펙 (6-inch Wafer, WLP)
# ============================================================
# Wafer: 6-inch (152mm diameter)
# Die size: 5×5 mm
# Bump pitch: 150~300 µm
# Bump diameter: 90~100 µm
# Bump height: 60~80 µm (CPB)
# Bumps per die: 14×14 = 196
# Dies per wafer: ~700
# Total bumps per wafer: ~1.3~1.5×10⁵
# ============================================================


@dataclass
class MaterialProperties:
    """
    X-ray material properties based on NIST XCOM database

    Attributes:
        name: Material name
        density: Density in g/cm³
        Z_eff: Effective atomic number
        K_edge: K-edge energy in keV (if applicable)
    """
    name: str
    density: float  # g/cm³
    Z_eff: float    # Effective atomic number
    K_edge: Optional[float] = None  # keV


@dataclass
class XraySourceParams:
    """
    X-ray source parameters

    Attributes:
        kVp: Peak voltage (keV)
        mA: Tube current (mA)
        exposure_ms: Exposure time (ms)
        target: Anode material
        filter_material: Filter material
        filter_thickness_mm: Filter thickness (mm)
    """
    kVp: float = 80.0
    mA: float = 0.15
    exposure_ms: float = 50.0
    target: str = "W"  # Tungsten
    filter_material: str = "Al"
    filter_thickness_mm: float = 0.5


@dataclass
class DetectorParams:
    """
    X-ray detector parameters

    Attributes:
        pixel_size_um: Pixel size (µm)
        quantum_efficiency: QE at operating energy
        gain: Detector gain (electrons/photon)
        readout_noise: Readout noise std (electrons)
        dark_current: Dark current (electrons/pixel/s)
        bit_depth: ADC bit depth
    """
    pixel_size_um: float = 5.0
    quantum_efficiency: float = 0.85
    gain: float = 100.0
    readout_noise: float = 5.0
    dark_current: float = 0.1
    bit_depth: int = 16


@dataclass
class BumpGeometry:
    """
    Solder bump geometry parameters (project spec)

    Attributes:
        diameter_um: Bump diameter (µm)
        height_um: Bump height (µm)
        pitch_um: Bump pitch (µm)
        ubm_thickness_um: UBM layer thickness (µm)
    """
    diameter_um: float = 100.0
    height_um: float = 70.0
    pitch_um: float = 150.0
    ubm_thickness_um: float = 5.0


class MaterialDatabase:
    """
    Material database with X-ray attenuation coefficients
    Based on NIST XCOM database
    """

    def __init__(self):
        self.materials = {
            # SAC305 (Sn96.5Ag3Cu0.5) - Lead-free solder
            "SAC305": MaterialProperties(
                name="SAC305",
                density=7.4,
                Z_eff=48.5,
                K_edge=29.2  # Sn K-edge
            ),
            # Copper (UBM layer)
            "Cu": MaterialProperties(
                name="Copper",
                density=8.96,
                Z_eff=29.0,
                K_edge=8.98
            ),
            # Silicon (substrate)
            "Si": MaterialProperties(
                name="Silicon",
                density=2.33,
                Z_eff=14.0,
                K_edge=1.84
            ),
            # Air/Void
            "Air": MaterialProperties(
                name="Air",
                density=0.001225,
                Z_eff=7.5,
                K_edge=None
            ),
            # Aluminum (filter)
            "Al": MaterialProperties(
                name="Aluminum",
                density=2.70,
                Z_eff=13.0,
                K_edge=1.56
            ),
            # Tungsten (X-ray target)
            "W": MaterialProperties(
                name="Tungsten",
                density=19.3,
                Z_eff=74.0,
                K_edge=69.5
            )
        }

    def get_mass_attenuation_coefficient(self, material: str, energy_keV: float) -> float:
        """
        Calculate mass attenuation coefficient (cm²/g)
        Using simplified power-law approximation of NIST data

        μ/ρ ≈ C × Z^n × E^(-m)

        where C, n, m are empirical constants
        """
        mat = self.materials.get(material)
        if mat is None:
            return 0.0

        Z = mat.Z_eff
        E = max(energy_keV, 1.0)  # Avoid division by zero

        # Power-law approximation (valid for 10-100 keV range)
        # Based on photoelectric absorption dominance
        if mat.K_edge and E > mat.K_edge:
            # Above K-edge: absorption increases
            mu_rho = 0.5 * (Z / 20)**3.5 * (80 / E)**2.8
        else:
            # Below K-edge
            mu_rho = 0.3 * (Z / 20)**3.2 * (80 / E)**2.5

        return mu_rho

    def get_linear_attenuation_coefficient(self, material: str, energy_keV: float) -> float:
        """
        Calculate linear attenuation coefficient (cm⁻¹)
        μ = (μ/ρ) × ρ
        """
        mat = self.materials.get(material)
        if mat is None:
            return 0.0

        mu_rho = self.get_mass_attenuation_coefficient(material, energy_keV)
        return mu_rho * mat.density


class XraySpectrum:
    """
    X-ray spectrum generator
    Based on Kramers' law with characteristic X-rays
    """

    def __init__(self, source_params: XraySourceParams):
        self.params = source_params
        self.material_db = MaterialDatabase()

    def generate_spectrum(self, num_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate X-ray spectrum

        Returns:
            energies: Energy bins (keV)
            intensities: Normalized intensity distribution
        """
        kVp = self.params.kVp
        energies = np.linspace(1, kVp, num_bins)

        # Bremsstrahlung (Kramers' law)
        # I(E) ∝ Z × (E_max - E) / E
        bremsstrahlung = np.maximum(0, (kVp - energies)) / (energies + 0.1)
        bremsstrahlung *= np.exp(-energies / 30)  # Self-absorption correction

        # Characteristic X-rays (Tungsten target)
        characteristic = np.zeros_like(energies)
        if self.params.target == "W":
            # W Kα1: 59.3 keV, Kα2: 58.0 keV, Kβ1: 67.2 keV
            if kVp > 59.3:
                idx_ka = np.argmin(np.abs(energies - 59.3))
                characteristic[idx_ka] = 0.3 * np.max(bremsstrahlung)
            if kVp > 67.2:
                idx_kb = np.argmin(np.abs(energies - 67.2))
                characteristic[idx_kb] = 0.15 * np.max(bremsstrahlung)

        spectrum = bremsstrahlung + characteristic

        # Apply filter attenuation
        filter_atten = self._apply_filter(energies)
        spectrum *= filter_atten

        # Normalize
        spectrum = spectrum / (np.sum(spectrum) + 1e-10)

        return energies, spectrum

    def _apply_filter(self, energies: np.ndarray) -> np.ndarray:
        """Apply filter attenuation"""
        mat = self.params.filter_material
        thickness_cm = self.params.filter_thickness_mm / 10.0

        attenuation = np.zeros_like(energies)
        for i, E in enumerate(energies):
            mu = self.material_db.get_linear_attenuation_coefficient(mat, E)
            attenuation[i] = np.exp(-mu * thickness_cm)

        return attenuation


class PhysicsXraySimulator:
    """
    Physics-based X-ray simulator with Beer-Lambert law
    Optimized with NumPy vectorization
    """

    def __init__(self,
                 source_params: Optional[XraySourceParams] = None,
                 detector_params: Optional[DetectorParams] = None,
                 bump_geometry: Optional[BumpGeometry] = None):

        self.source = source_params or XraySourceParams()
        self.detector = detector_params or DetectorParams()
        self.bump_geom = bump_geometry or BumpGeometry()

        self.material_db = MaterialDatabase()
        self.spectrum_gen = XraySpectrum(self.source)

        # Pre-compute spectrum
        self.energies, self.spectrum = self.spectrum_gen.generate_spectrum(50)
        self.mean_energy = np.sum(self.energies * self.spectrum)

    def generate_bump_projection(self,
                                  size: int = 64,
                                  defect_type: str = "Normal",
                                  defect_params: Optional[Dict] = None) -> np.ndarray:
        """
        Generate single bump X-ray projection image

        Args:
            size: Image size (pixels)
            defect_type: Type of defect
            defect_params: Optional defect parameters

        Returns:
            8-bit grayscale image
        """
        # Create coordinate grids (vectorized)
        y, x = np.ogrid[:size, :size]
        center = size / 2

        # Bump geometry in pixels
        pixel_size_um = self.bump_geom.diameter_um / (size * 0.6)  # 60% fill
        radius_px = self.bump_geom.diameter_um / (2 * pixel_size_um)
        height_um = self.bump_geom.height_um

        # Distance from center
        dist = np.sqrt((x - center)**2 + (y - center)**2)

        # Spherical cap thickness profile (vectorized)
        # t(r) = 2 × sqrt(R² - r²) for |r| < R
        bump_mask = dist < radius_px
        thickness_px = np.zeros((size, size), dtype=np.float32)
        thickness_px[bump_mask] = 2 * np.sqrt(
            np.maximum(0, radius_px**2 - dist[bump_mask]**2)
        )

        # Convert to physical thickness (µm → cm)
        thickness_cm = thickness_px * pixel_size_um / 10000

        # Apply defect modifications
        if defect_type != "Normal":
            thickness_cm = self._apply_defect(
                thickness_cm, defect_type, center, radius_px, defect_params
            )

        # Beer-Lambert attenuation (polychromatic)
        transmission = self._calculate_transmission(thickness_cm, "SAC305")

        # Apply detector response
        image = self._apply_detector_response(transmission)

        return image

    def _calculate_transmission(self, thickness_cm: np.ndarray, material: str) -> np.ndarray:
        """
        Calculate X-ray transmission using Beer-Lambert law
        I/I₀ = exp(-μt)

        For polychromatic beam, integrate over spectrum
        """
        transmission = np.zeros_like(thickness_cm)

        for i, (E, weight) in enumerate(zip(self.energies, self.spectrum)):
            if weight < 1e-6:
                continue
            mu = self.material_db.get_linear_attenuation_coefficient(material, E)
            transmission += weight * np.exp(-mu * thickness_cm)

        return transmission

    def _apply_defect(self, thickness: np.ndarray, defect_type: str,
                      center: float, radius: float,
                      params: Optional[Dict] = None) -> np.ndarray:
        """Apply defect to thickness map"""

        size = thickness.shape[0]
        y, x = np.ogrid[:size, :size]

        if defect_type == "Void":
            # Internal voids (gas bubbles)
            num_voids = random.randint(2, 5)
            for _ in range(num_voids):
                void_r = random.uniform(2, 6)
                offset_x = random.uniform(-radius*0.5, radius*0.5)
                offset_y = random.uniform(-radius*0.5, radius*0.5)
                void_cx = center + offset_x
                void_cy = center + offset_y

                void_dist = np.sqrt((x - void_cx)**2 + (y - void_cy)**2)
                void_mask = void_dist < void_r

                # Void reduces effective thickness
                void_depth = np.sqrt(np.maximum(0, void_r**2 - void_dist**2)) * 2
                thickness[void_mask] -= void_depth[void_mask] * 0.5

            thickness = np.maximum(thickness, 0)

        elif defect_type == "Bridge":
            # Solder bridge between adjacent bumps
            bridge_width = random.randint(3, 8)
            bridge_y_start = int(center - bridge_width // 2)
            bridge_y_end = int(center + bridge_width // 2)

            bridge_mask = (y >= bridge_y_start) & (y < bridge_y_end) & (x > center + radius - 3)
            thickness[bridge_mask] = np.mean(thickness[thickness > 0]) * 0.3

        elif defect_type == "HiP":
            # Head-in-Pillow: partial non-wetting
            hip_mask = (y > center) & (np.sqrt((x-center)**2 + (y-center)**2) < radius)
            thickness[hip_mask] *= random.uniform(0.3, 0.6)

        elif defect_type == "ColdJoint":
            # Cold joint: irregular surface, incomplete reflow
            noise = np.random.normal(0, 0.15, thickness.shape)
            bump_mask = thickness > 0
            thickness[bump_mask] += thickness[bump_mask] * noise[bump_mask]
            thickness = np.maximum(thickness, 0)

        elif defect_type == "Crack":
            # Crack: linear discontinuity
            angle = random.uniform(0, np.pi)
            crack_width = 2

            # Line equation: y - cy = tan(angle) * (x - cx)
            dist_to_line = np.abs(
                (y - center) * np.cos(angle) - (x - center) * np.sin(angle)
            )
            crack_mask = (dist_to_line < crack_width) & (thickness > 0)
            thickness[crack_mask] *= random.uniform(0.2, 0.4)

        return thickness

    def _apply_detector_response(self, transmission: np.ndarray) -> np.ndarray:
        """
        Apply detector response model

        1. Photon count from transmission
        2. Poisson noise (shot noise)
        3. Gaussian noise (readout noise)
        4. ADC conversion
        """
        # Incident photon flux (photons/pixel)
        flux = self.source.mA * self.source.exposure_ms * 1e6  # Arbitrary scaling

        # Detected photons
        detected = transmission * flux * self.detector.quantum_efficiency

        # Poisson noise (shot noise) - vectorized
        # Using normal approximation for large counts
        detected_noisy = detected + np.sqrt(np.maximum(detected, 1)) * np.random.randn(*detected.shape)
        detected_noisy = np.maximum(detected_noisy, 0)

        # Convert to electrons
        electrons = detected_noisy * self.detector.gain

        # Add readout noise (Gaussian)
        electrons += np.random.normal(0, self.detector.readout_noise, electrons.shape)

        # Add dark current
        dark = self.detector.dark_current * self.source.exposure_ms / 1000
        electrons += dark

        # ADC conversion
        max_val = 2**self.detector.bit_depth - 1

        # Normalize to 0-255 (8-bit output)
        image = electrons / (np.max(electrons) + 1e-10) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def generate_die_image(self,
                           grid_size: Tuple[int, int] = (14, 14),
                           bump_size: int = 40,
                           padding: int = 6,
                           defect_distribution: Optional[Dict[str, float]] = None
                           ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate die-level X-ray image with multiple bumps

        Args:
            grid_size: Bump array size (rows, cols)
            bump_size: Individual bump image size (pixels)
            padding: Padding between bumps (pixels)
            defect_distribution: Defect type probabilities

        Returns:
            image: Die X-ray image
            annotations: YOLO format annotations
        """
        if defect_distribution is None:
            defect_distribution = {
                "Normal": 0.70,
                "Void": 0.12,
                "Bridge": 0.06,
                "HiP": 0.04,
                "ColdJoint": 0.04,
                "Crack": 0.04
            }

        rows, cols = grid_size
        img_width = cols * bump_size + (cols + 1) * padding
        img_height = rows * bump_size + (rows + 1) * padding

        # Background (substrate)
        die_image = np.ones((img_height, img_width), dtype=np.uint8) * 180

        annotations = []
        total_bumps = rows * cols

        # Assign defects based on distribution
        defect_assignments = self._assign_defects(total_bumps, defect_distribution)

        bump_idx = 0
        for row in range(rows):
            for col in range(cols):
                defect_type = defect_assignments[bump_idx]

                # Generate individual bump
                bump_img = self.generate_bump_projection(bump_size, defect_type)

                # Position
                x = padding + col * (bump_size + padding)
                y = padding + row * (bump_size + padding)

                # Place bump
                die_image[y:y+bump_size, x:x+bump_size] = bump_img

                # YOLO annotation (normalized coordinates)
                class_id = self._get_class_id(defect_type)
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
                    "height": height
                })

                bump_idx += 1

        return die_image, annotations

    def _assign_defects(self, num_bumps: int, distribution: Dict[str, float]) -> List[str]:
        """Assign defects based on probability distribution"""
        defects = []
        for defect_type, prob in distribution.items():
            count = int(num_bumps * prob)
            defects.extend([defect_type] * count)

        # Fill remaining with Normal
        while len(defects) < num_bumps:
            defects.append("Normal")

        random.shuffle(defects)
        return defects[:num_bumps]

    def _get_class_id(self, defect_type: str) -> int:
        """Get class ID for defect type"""
        class_map = {
            "Void": 0,
            "Bridge": 1,
            "HiP": 2,
            "ColdJoint": 3,
            "Crack": 4,
            "Normal": 5
        }
        return class_map.get(defect_type, 5)


class WaferLevelSimulator:
    """
    Full wafer-level X-ray image generator
    6-inch wafer with circular die arrangement

    Specs:
    - Wafer diameter: 152mm (6-inch)
    - Die size: 5×5mm
    - Bumps per die: 14×14 = 196
    - Dies per wafer: ~700
    - Total bumps: ~137,000
    """

    def __init__(self,
                 wafer_diameter_mm: float = 152.0,
                 die_size_mm: float = 5.0,
                 edge_exclusion_mm: float = 3.0):
        """
        Initialize wafer simulator

        Args:
            wafer_diameter_mm: Wafer diameter in mm
            die_size_mm: Die size in mm (square)
            edge_exclusion_mm: Edge exclusion zone in mm
        """
        self.wafer_diameter = wafer_diameter_mm
        self.die_size = die_size_mm
        self.edge_exclusion = edge_exclusion_mm
        self.wafer_radius = wafer_diameter_mm / 2 - edge_exclusion_mm

        # Die simulator
        self.die_simulator = PhysicsXraySimulator()

        # Calculate valid die positions
        self.die_positions = self._calculate_die_positions()

    def _calculate_die_positions(self) -> List[Tuple[int, int, float, float]]:
        """
        Calculate which die positions are valid (inside wafer circle)

        Returns:
            List of (grid_x, grid_y, center_x_mm, center_y_mm)
        """
        positions = []

        # Grid dimensions
        num_dies_per_side = int(self.wafer_diameter / self.die_size) + 1
        start = -num_dies_per_side // 2
        end = num_dies_per_side // 2 + 1

        for gy in range(start, end):
            for gx in range(start, end):
                # Die center position (mm from wafer center)
                cx = gx * self.die_size
                cy = gy * self.die_size

                # Check all 4 corners of die
                corners_valid = True
                for dx in [-0.5, 0.5]:
                    for dy in [-0.5, 0.5]:
                        corner_x = cx + dx * self.die_size
                        corner_y = cy + dy * self.die_size
                        dist = np.sqrt(corner_x**2 + corner_y**2)
                        if dist > self.wafer_radius:
                            corners_valid = False
                            break
                    if not corners_valid:
                        break

                if corners_valid:
                    positions.append((gx, gy, cx, cy))

        return positions

    def get_die_count(self) -> int:
        """Get number of valid dies"""
        return len(self.die_positions)

    def get_total_bumps(self, bumps_per_die: int = 196) -> int:
        """Get total number of bumps on wafer"""
        return self.get_die_count() * bumps_per_die

    def generate_wafer_image(self,
                             image_size: int = 4096,
                             die_bump_grid: Tuple[int, int] = (14, 14),
                             bump_pixel_size: int = 8,
                             defect_distribution: Optional[Dict[str, float]] = None
                             ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate full wafer X-ray image

        Args:
            image_size: Output image size (pixels, square)
            die_bump_grid: Bumps per die (rows, cols)
            bump_pixel_size: Pixel size per bump in output
            defect_distribution: Defect probabilities

        Returns:
            wafer_image: Full wafer X-ray image
            all_annotations: YOLO format annotations for all bumps
        """
        if defect_distribution is None:
            defect_distribution = {
                "Normal": 0.70,
                "Void": 0.12,
                "Bridge": 0.06,
                "HiP": 0.04,
                "ColdJoint": 0.04,
                "Crack": 0.04
            }

        # Create wafer background (gray for silicon, darker outside)
        wafer_image = np.zeros((image_size, image_size), dtype=np.uint8)

        # Scale factor: mm to pixels
        scale = image_size / self.wafer_diameter
        center = image_size // 2

        # Draw wafer circle (silicon substrate)
        wafer_radius_px = int((self.wafer_diameter / 2) * scale)
        cv2.circle(wafer_image, (center, center), wafer_radius_px, 180, -1)

        # Die size in pixels
        die_size_px = int(self.die_size * scale)
        bump_rows, bump_cols = die_bump_grid
        bumps_per_die = bump_rows * bump_cols

        # Calculate padding and bump size for each die
        bump_size_in_die = die_size_px // max(bump_rows, bump_cols)
        padding_in_die = (die_size_px - bump_size_in_die * bump_cols) // (bump_cols + 1)

        all_annotations = []
        total_dies = len(self.die_positions)

        print(f"\nGenerating wafer image with {total_dies} dies...")
        print(f"  - Image size: {image_size}×{image_size} pixels")
        print(f"  - Die size: {die_size_px}×{die_size_px} pixels")
        print(f"  - Bumps per die: {bumps_per_die}")
        print(f"  - Total bumps: {total_dies * bumps_per_die:,}")

        for die_idx, (gx, gy, cx_mm, cy_mm) in enumerate(tqdm(self.die_positions, desc="  Dies")):
            # Die top-left corner in pixels
            die_x = center + int((cx_mm - self.die_size/2) * scale)
            die_y = center + int((cy_mm - self.die_size/2) * scale)

            # Assign defects for this die
            defect_assignments = self._assign_defects(bumps_per_die, defect_distribution)

            bump_idx = 0
            for row in range(bump_rows):
                for col in range(bump_cols):
                    defect_type = defect_assignments[bump_idx]

                    # Bump position within die
                    bx = padding_in_die + col * (bump_size_in_die + padding_in_die)
                    by = padding_in_die + row * (bump_size_in_die + padding_in_die)

                    # Absolute position
                    abs_x = die_x + bx
                    abs_y = die_y + by

                    # Generate and place bump
                    if bump_size_in_die >= 8:
                        bump_img = self.die_simulator.generate_bump_projection(
                            bump_size_in_die, defect_type
                        )
                        # Ensure within bounds
                        if (0 <= abs_y < image_size - bump_size_in_die and
                            0 <= abs_x < image_size - bump_size_in_die):
                            wafer_image[abs_y:abs_y+bump_size_in_die,
                                       abs_x:abs_x+bump_size_in_die] = bump_img
                    else:
                        # Too small, just draw circle
                        bump_center = (abs_x + bump_size_in_die//2,
                                      abs_y + bump_size_in_die//2)
                        radius = max(1, bump_size_in_die // 2 - 1)
                        intensity = 60 if defect_type == "Normal" else 100
                        if defect_type == "Void":
                            intensity = 140
                        cv2.circle(wafer_image, bump_center, radius, intensity, -1)

                    # YOLO annotation
                    x_center = (abs_x + bump_size_in_die/2) / image_size
                    y_center = (abs_y + bump_size_in_die/2) / image_size
                    width = bump_size_in_die / image_size
                    height = bump_size_in_die / image_size

                    class_id = self._get_class_id(defect_type)
                    all_annotations.append({
                        "class_id": class_id,
                        "class_name": defect_type,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "die_idx": die_idx,
                        "bump_row": row,
                        "bump_col": col
                    })

                    bump_idx += 1

        # Add wafer notch (orientation mark)
        notch_y = center + wafer_radius_px - 5
        cv2.circle(wafer_image, (center, notch_y), 8, 0, -1)

        return wafer_image, all_annotations

    def _assign_defects(self, num_bumps: int, distribution: Dict[str, float]) -> List[str]:
        """Assign defects based on probability distribution"""
        defects = []
        for defect_type, prob in distribution.items():
            count = int(num_bumps * prob)
            defects.extend([defect_type] * count)

        while len(defects) < num_bumps:
            defects.append("Normal")

        random.shuffle(defects)
        return defects[:num_bumps]

    def _get_class_id(self, defect_type: str) -> int:
        """Get class ID for defect type"""
        class_map = {
            "Void": 0, "Bridge": 1, "HiP": 2,
            "ColdJoint": 3, "Crack": 4, "Normal": 5
        }
        return class_map.get(defect_type, 5)


class PhysicsDatasetGenerator:
    """
    Dataset generator using physics-based simulation
    Supports both die-level and wafer-level images
    """

    CLASS_NAMES = {
        0: "Void",
        1: "Bridge",
        2: "HiP",
        3: "ColdJoint",
        4: "Crack",
        5: "Normal"
    }

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.simulator = PhysicsXraySimulator()

    def generate_dataset(self,
                        num_images: int = 500,
                        grid_size: Tuple[int, int] = (14, 14),
                        bump_size: int = 40,
                        padding: int = 6,
                        train_ratio: float = 0.7,
                        valid_ratio: float = 0.2):
        """Generate complete dataset"""

        # Create directories
        for split in ["train", "valid", "test"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Calculate splits
        train_count = int(num_images * train_ratio)
        valid_count = int(num_images * valid_ratio)
        test_count = num_images - train_count - valid_count

        bumps_per_image = grid_size[0] * grid_size[1]

        print("=" * 60)
        print("Physics-Based X-ray Dataset Generation")
        print("=" * 60)
        print(f"Physical Models:")
        print(f"  - Beer-Lambert Law with polychromatic spectrum")
        print(f"  - Poisson + Gaussian noise model")
        print(f"  - X-ray source: {self.simulator.source.kVp} kVp")
        print(f"  - Mean energy: {self.simulator.mean_energy:.1f} keV")
        print(f"Dataset:")
        print(f"  - Total images: {num_images}")
        print(f"  - Train/Valid/Test: {train_count}/{valid_count}/{test_count}")
        print(f"  - Bumps per image: {bumps_per_image}")
        print(f"  - Total bumps: {num_images * bumps_per_image:,}")
        print("=" * 60)

        splits = [
            ("train", train_count),
            ("valid", valid_count),
            ("test", test_count)
        ]

        stats = {name: 0 for name in self.CLASS_NAMES.values()}

        for split_name, count in splits:
            print(f"\n[{split_name}] Generating {count} images...")

            for i in tqdm(range(count), desc=f"  {split_name}"):
                # Generate image
                img, annotations = self.simulator.generate_die_image(
                    grid_size, bump_size, padding
                )

                # Save image
                filename = f"{split_name}_{i:06d}"
                img_path = self.output_dir / split_name / "images" / f"{filename}.png"
                cv2.imwrite(str(img_path), img)

                # Save labels
                label_path = self.output_dir / split_name / "labels" / f"{filename}.txt"
                with open(label_path, "w") as f:
                    for ann in annotations:
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                        f.write(line)
                        stats[ann['class_name']] += 1

        # Create data.yaml
        self._create_data_yaml()

        # Print statistics
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"\nOutput: {self.output_dir}")
        print("\nClass distribution:")
        total = sum(stats.values())
        for name, count in stats.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {name:12s}: {count:6d} ({pct:5.1f}%)")
        print(f"  {'Total':12s}: {total:6d}")

    def _create_data_yaml(self):
        """Create data.yaml for YOLOv8"""
        data_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": self.CLASS_NAMES,
            "nc": len(self.CLASS_NAMES)
        }

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)


def test_wafer_level():
    """Test wafer-level generation"""
    print("=" * 60)
    print("Wafer-Level X-ray Simulation Test")
    print("=" * 60)

    wafer_sim = WaferLevelSimulator(
        wafer_diameter_mm=152.0,  # 6-inch
        die_size_mm=5.0,
        edge_exclusion_mm=3.0
    )

    print(f"\nWafer Configuration:")
    print(f"  - Diameter: 152mm (6-inch)")
    print(f"  - Die size: 5×5mm")
    print(f"  - Valid dies: {wafer_sim.get_die_count()}")
    print(f"  - Bumps per die: 196 (14×14)")
    print(f"  - Total bumps: {wafer_sim.get_total_bumps():,}")

    # Generate wafer image (4K resolution)
    wafer_img, annotations = wafer_sim.generate_wafer_image(
        image_size=4096,
        die_bump_grid=(14, 14)
    )

    # Save image
    output_path = Path(__file__).parent.parent / "data" / "sample_wafer_full.png"
    cv2.imwrite(str(output_path), wafer_img)
    print(f"\nSaved: {output_path}")

    # Statistics
    stats = {}
    for ann in annotations:
        name = ann["class_name"]
        stats[name] = stats.get(name, 0) + 1

    print(f"\nDefect Distribution:")
    for name, count in sorted(stats.items()):
        print(f"  {name}: {count:,}")

    return wafer_img, annotations


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Physics-Based X-ray Solder Bump Dataset Generator"
    )
    parser.add_argument("--mode", type=str, default="die",
                       choices=["die", "wafer", "both"],
                       help="Generation mode: die (die-level), wafer (wafer-level), both")
    parser.add_argument("--num_images", type=int, default=500,
                       help="Number of images to generate")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[14, 14],
                       help="Bump array size (rows cols)")
    parser.add_argument("--wafer_size", type=int, default=4096,
                       help="Wafer image size in pixels")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--test", action="store_true",
                       help="Run test mode (single wafer image)")

    args = parser.parse_args()

    if args.test:
        test_wafer_level()
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "data" / "xray"

    if args.mode == "wafer":
        # Wafer-level dataset generation
        print("=" * 60)
        print("Wafer-Level Dataset Generation")
        print("=" * 60)

        wafer_sim = WaferLevelSimulator()
        print(f"\nWafer specs:")
        print(f"  - Dies per wafer: {wafer_sim.get_die_count()}")
        print(f"  - Bumps per wafer: {wafer_sim.get_total_bumps():,}")

        # Create directories
        for split in ["train", "valid", "test"]:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        train_count = int(args.num_images * 0.7)
        valid_count = int(args.num_images * 0.2)
        test_count = args.num_images - train_count - valid_count

        splits = [("train", train_count), ("valid", valid_count), ("test", test_count)]

        for split_name, count in splits:
            print(f"\n[{split_name}] Generating {count} wafer images...")
            for i in tqdm(range(count), desc=f"  {split_name}"):
                wafer_img, annotations = wafer_sim.generate_wafer_image(
                    image_size=args.wafer_size,
                    die_bump_grid=tuple(args.grid_size)
                )

                filename = f"{split_name}_{i:06d}"
                img_path = output_dir / split_name / "images" / f"{filename}.png"
                cv2.imwrite(str(img_path), wafer_img)

                label_path = output_dir / split_name / "labels" / f"{filename}.txt"
                with open(label_path, "w") as f:
                    for ann in annotations:
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                        f.write(line)

        # Create data.yaml
        PhysicsDatasetGenerator(output_dir)._create_data_yaml()
        print(f"\nDataset saved to: {output_dir}")

    else:
        # Die-level (original mode)
        generator = PhysicsDatasetGenerator(output_dir)
        generator.generate_dataset(
            num_images=args.num_images,
            grid_size=tuple(args.grid_size)
        )


if __name__ == "__main__":
    main()
