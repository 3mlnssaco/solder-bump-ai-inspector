#!/usr/bin/env python3
"""
솔더범프 X-ray 결함 검출 실시간 데모 GUI
Solder Bump X-ray Defect Detection Real-time Demo

Features:
- 이미지 업로드 및 실시간 검출
- 결함 유형별 색상 표시
- 통계 대시보드
- 웨이퍼 맵 시각화
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# PyQt6 import
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                  QHBoxLayout, QLabel, QPushButton, QFileDialog,
                                  QGroupBox, QGridLayout, QProgressBar, QTableWidget,
                                  QTableWidgetItem, QHeaderView, QSplitter, QFrame,
                                  QComboBox, QSpinBox, QCheckBox, QTextEdit, QTabWidget)
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                      QHBoxLayout, QLabel, QPushButton, QFileDialog,
                                      QGroupBox, QGridLayout, QProgressBar, QTableWidget,
                                      QTableWidgetItem, QHeaderView, QSplitter, QFrame,
                                      QComboBox, QSpinBox, QCheckBox, QTextEdit, QTabWidget)
        from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
        from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
        PYQT_VERSION = 5
    except ImportError:
        print("PyQt5 또는 PyQt6이 필요합니다.")
        print("설치: pip install PyQt6")
        sys.exit(1)

# YOLO 모델 로드
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Using simulation mode.")


class DefectDetector:
    """결함 검출 엔진"""

    CLASS_NAMES = {0: 'Void', 1: 'Bridge', 2: 'HiP', 3: 'ColdJoint', 4: 'Crack', 5: 'Normal'}
    CLASS_COLORS = {
        0: (0, 0, 255),      # Void - 빨강
        1: (0, 165, 255),    # Bridge - 주황
        2: (0, 255, 255),    # HiP - 노랑
        3: (255, 0, 255),    # ColdJoint - 마젠타
        4: (255, 0, 0),      # Crack - 파랑
        5: (0, 255, 0)       # Normal - 초록
    }

    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False

        if YOLO_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self.model = YOLO(model_path)
                self.model_loaded = True
                print(f"Model loaded: {model_path}")
            except Exception as e:
                print(f"Model loading failed: {e}")

    def detect(self, image, conf_threshold=0.5):
        """이미지에서 결함 검출"""
        results = {
            'detections': [],
            'stats': {name: 0 for name in self.CLASS_NAMES.values()},
            'total': 0,
            'defect_count': 0
        }

        if self.model_loaded and self.model is not None:
            # 실제 YOLO 추론
            preds = self.model(image, conf=conf_threshold, verbose=False)

            for pred in preds:
                boxes = pred.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detection = {
                        'class_id': cls_id,
                        'class_name': self.CLASS_NAMES.get(cls_id, 'Unknown'),
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    }
                    results['detections'].append(detection)
                    results['stats'][detection['class_name']] += 1
                    results['total'] += 1
                    if cls_id != 5:  # Not Normal
                        results['defect_count'] += 1
        else:
            # 시뮬레이션 모드
            results = self._simulate_detection(image)

        return results

    def _simulate_detection(self, image):
        """모델 없을 때 시뮬레이션"""
        h, w = image.shape[:2]
        results = {
            'detections': [],
            'stats': {name: 0 for name in self.CLASS_NAMES.values()},
            'total': 0,
            'defect_count': 0
        }

        # 14x14 그리드 가정
        grid_size = 14
        cell_w = w // grid_size
        cell_h = h // grid_size

        np.random.seed(int(datetime.now().timestamp()) % 1000)

        for row in range(grid_size):
            for col in range(grid_size):
                # 랜덤 클래스 할당 (70% Normal)
                rand = np.random.random()
                if rand < 0.70:
                    cls_id = 5  # Normal
                elif rand < 0.82:
                    cls_id = 0  # Void
                elif rand < 0.88:
                    cls_id = 1  # Bridge
                elif rand < 0.92:
                    cls_id = 2  # HiP
                elif rand < 0.96:
                    cls_id = 3  # ColdJoint
                else:
                    cls_id = 4  # Crack

                x1 = col * cell_w + 2
                y1 = row * cell_h + 2
                x2 = x1 + cell_w - 4
                y2 = y1 + cell_h - 4

                conf = np.random.uniform(0.75, 0.98)

                detection = {
                    'class_id': cls_id,
                    'class_name': self.CLASS_NAMES[cls_id],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                }
                results['detections'].append(detection)
                results['stats'][detection['class_name']] += 1
                results['total'] += 1
                if cls_id != 5:
                    results['defect_count'] += 1

        return results

    def draw_detections(self, image, results, show_normal=False):
        """검출 결과 시각화"""
        output = image.copy()

        for det in results['detections']:
            cls_id = det['class_id']

            # Normal은 옵션에 따라 표시
            if cls_id == 5 and not show_normal:
                continue

            x1, y1, x2, y2 = det['bbox']
            color = self.CLASS_COLORS[cls_id]
            conf = det['confidence']

            # 박스 그리기
            thickness = 2 if cls_id != 5 else 1
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # 라벨
            if cls_id != 5:  # 결함만 라벨 표시
                label = f"{det['class_name']} {conf:.2f}"
                font_scale = 0.4
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(output, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
                cv2.putText(output, label, (x1+2, y1-2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)

        return output


class DetectionThread(QThread):
    """백그라운드 검출 스레드"""
    finished = pyqtSignal(dict)

    def __init__(self, detector, image, conf_threshold):
        super().__init__()
        self.detector = detector
        self.image = image
        self.conf_threshold = conf_threshold

    def run(self):
        results = self.detector.detect(self.image, self.conf_threshold)
        self.finished.emit(results)


class MainWindow(QMainWindow):
    """메인 GUI 윈도우"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("솔더범프 X-ray 결함 검출 시스템 v1.0")
        self.setGeometry(100, 100, 1400, 900)

        # 모델 경로
        self.model_path = Path(__file__).parent / "runs/xray_detection/physics_resume/weights/best.pt"
        self.detector = DefectDetector(str(self.model_path))

        self.current_image = None
        self.current_results = None

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """UI 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # 왼쪽: 이미지 뷰어
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 탭 위젯
        self.tab_widget = QTabWidget()

        # 탭 1: 원본 이미지
        self.original_label = QLabel("이미지를 로드하세요")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(600, 500)
        self.original_label.setStyleSheet("border: 2px solid #3498db; background: #2c3e50;")
        self.tab_widget.addTab(self.original_label, "원본 이미지")

        # 탭 2: 검출 결과
        self.result_label = QLabel("검출 결과가 여기에 표시됩니다")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumSize(600, 500)
        self.result_label.setStyleSheet("border: 2px solid #e74c3c; background: #2c3e50;")
        self.tab_widget.addTab(self.result_label, "검출 결과")

        left_layout.addWidget(self.tab_widget)

        # 버튼 영역
        btn_layout = QHBoxLayout()

        self.load_btn = QPushButton("📁 이미지 로드")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setMinimumHeight(40)

        self.detect_btn = QPushButton("🔍 결함 검출")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setMinimumHeight(40)
        self.detect_btn.setEnabled(False)

        self.sample_btn = QPushButton("📷 샘플 로드")
        self.sample_btn.clicked.connect(self.load_sample)
        self.sample_btn.setMinimumHeight(40)

        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.detect_btn)
        btn_layout.addWidget(self.sample_btn)

        left_layout.addLayout(btn_layout)

        # 오른쪽: 결과 패널
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 설정 그룹
        settings_group = QGroupBox("검출 설정")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Confidence 임계값:"), 0, 0)
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(10, 95)
        self.conf_spin.setValue(50)
        self.conf_spin.setSuffix("%")
        settings_layout.addWidget(self.conf_spin, 0, 1)

        self.show_normal_check = QCheckBox("Normal 표시")
        self.show_normal_check.setChecked(False)
        self.show_normal_check.stateChanged.connect(self.update_display)
        settings_layout.addWidget(self.show_normal_check, 1, 0, 1, 2)

        settings_group.setLayout(settings_layout)
        right_layout.addWidget(settings_group)

        # 통계 그룹
        stats_group = QGroupBox("검출 통계")
        stats_layout = QVBoxLayout()

        # 요약 라벨
        self.summary_label = QLabel("총 범프: -\n결함 수: -\n양품률: -")
        self.summary_label.setFont(QFont("Arial", 12))
        stats_layout.addWidget(self.summary_label)

        # 클래스별 테이블
        self.stats_table = QTableWidget(6, 3)
        self.stats_table.setHorizontalHeaderLabels(["결함 유형", "개수", "비율"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.setMaximumHeight(200)

        class_names = ['Void', 'Bridge', 'HiP', 'ColdJoint', 'Crack', 'Normal']
        for i, name in enumerate(class_names):
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem("-"))
            self.stats_table.setItem(i, 2, QTableWidgetItem("-"))

        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        # 로그 영역
        log_group = QGroupBox("검출 로그")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        # 범례
        legend_group = QGroupBox("범례")
        legend_layout = QGridLayout()

        colors = [
            ("Void", "#FF0000"),
            ("Bridge", "#FFA500"),
            ("HiP", "#FFFF00"),
            ("ColdJoint", "#FF00FF"),
            ("Crack", "#0000FF"),
            ("Normal", "#00FF00")
        ]

        for i, (name, color) in enumerate(colors):
            color_label = QLabel("■")
            color_label.setStyleSheet(f"color: {color}; font-size: 16px;")
            legend_layout.addWidget(color_label, i // 3, (i % 3) * 2)
            legend_layout.addWidget(QLabel(name), i // 3, (i % 3) * 2 + 1)

        legend_group.setLayout(legend_layout)
        right_layout.addWidget(legend_group)

        right_layout.addStretch()

        # 메인 레이아웃에 추가
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 400])

        main_layout.addWidget(splitter)

        # 상태바
        self.statusBar().showMessage("준비됨 | 모델: " +
            ("로드됨 ✓" if self.detector.model_loaded else "시뮬레이션 모드"))

    def _apply_style(self):
        """스타일 적용"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ecf0f1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
            QLabel {
                color: #ecf0f1;
            }
            QTableWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
                gridline-color: #34495e;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: #ecf0f1;
                padding: 5px;
                border: 1px solid #2c3e50;
            }
            QTextEdit {
                background-color: #2c3e50;
                color: #2ecc71;
                border: 1px solid #34495e;
                font-family: Consolas, monospace;
            }
            QSpinBox, QComboBox {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                padding: 5px;
            }
            QCheckBox {
                color: #ecf0f1;
            }
            QTabWidget::pane {
                border: 1px solid #3498db;
            }
            QTabBar::tab {
                background-color: #34495e;
                color: #ecf0f1;
                padding: 8px 16px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
            }
        """)

    def load_image(self):
        """이미지 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self._display_image(self.current_image, self.original_label)
                self.detect_btn.setEnabled(True)
                self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 이미지 로드: {Path(file_path).name}")
                self.tab_widget.setCurrentIndex(0)

    def load_sample(self):
        """샘플 이미지 로드"""
        sample_paths = [
            Path(__file__).parent / "presentation_materials/sample_die_image.png",
            Path(__file__).parent / "data/sample_physics_die.png",
            Path(__file__).parent / "data/xray/valid/images/valid_000099.png"
        ]

        for path in sample_paths:
            if path.exists():
                self.current_image = cv2.imread(str(path))
                if self.current_image is not None:
                    self._display_image(self.current_image, self.original_label)
                    self.detect_btn.setEnabled(True)
                    self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 샘플 로드: {path.name}")
                    self.tab_widget.setCurrentIndex(0)
                    return

        # 샘플 없으면 생성
        self._generate_sample()

    def _generate_sample(self):
        """샘플 이미지 생성"""
        size = 650
        img = np.ones((size, size), dtype=np.uint8) * 180

        grid = 14
        cell = size // grid

        for row in range(grid):
            for col in range(grid):
                cx = col * cell + cell // 2
                cy = row * cell + cell // 2
                radius = cell // 3

                # 범프 그리기
                cv2.circle(img, (cx, cy), radius, 60, -1)
                cv2.circle(img, (cx, cy), radius, 40, 2)

        self.current_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self._display_image(self.current_image, self.original_label)
        self.detect_btn.setEnabled(True)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 샘플 이미지 생성됨")

    def run_detection(self):
        """결함 검출 실행"""
        if self.current_image is None:
            return

        self.detect_btn.setEnabled(False)
        self.statusBar().showMessage("검출 중...")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 검출 시작...")

        conf = self.conf_spin.value() / 100.0

        # 백그라운드 스레드에서 검출
        self.detection_thread = DetectionThread(self.detector, self.current_image, conf)
        self.detection_thread.finished.connect(self.on_detection_complete)
        self.detection_thread.start()

    def on_detection_complete(self, results):
        """검출 완료 처리"""
        self.current_results = results
        self.update_display()
        self.update_stats()

        self.detect_btn.setEnabled(True)
        self.statusBar().showMessage(f"검출 완료 | 총 {results['total']}개 범프, {results['defect_count']}개 결함")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 검출 완료: {results['defect_count']}개 결함 발견")

        self.tab_widget.setCurrentIndex(1)

    def update_display(self):
        """디스플레이 업데이트"""
        if self.current_image is None or self.current_results is None:
            return

        show_normal = self.show_normal_check.isChecked()
        result_image = self.detector.draw_detections(
            self.current_image, self.current_results, show_normal
        )
        self._display_image(result_image, self.result_label)

    def update_stats(self):
        """통계 업데이트"""
        if self.current_results is None:
            return

        results = self.current_results
        total = results['total']
        defects = results['defect_count']
        yield_rate = ((total - defects) / total * 100) if total > 0 else 0

        self.summary_label.setText(
            f"총 범프: {total}개\n"
            f"결함 수: {defects}개\n"
            f"양품률: {yield_rate:.1f}%"
        )

        # 테이블 업데이트
        class_names = ['Void', 'Bridge', 'HiP', 'ColdJoint', 'Crack', 'Normal']
        for i, name in enumerate(class_names):
            count = results['stats'].get(name, 0)
            ratio = (count / total * 100) if total > 0 else 0
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{ratio:.1f}%"))

    def _display_image(self, cv_image, label):
        """OpenCV 이미지를 QLabel에 표시"""
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # 레이블 크기에 맞게 스케일
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec() if PYQT_VERSION == 6 else app.exec_())


if __name__ == "__main__":
    main()
