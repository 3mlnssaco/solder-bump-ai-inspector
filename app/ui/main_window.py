"""
PyQt6 메인 윈도우
"""

import os
import sys
from typing import List, Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog,
    QProgressBar, QTextEdit, QTabWidget, QGroupBox,
    QDoubleSpinBox, QSpinBox, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

# 프로젝트 모듈
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector import DefectDetector
from utils.wafer_map import WaferMapVisualizer
from utils.report import ReportGenerator


class DetectionThread(QThread):
    """검출 작업 스레드"""
    
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list)  # results
    error = pyqtSignal(str)  # error message
    
    def __init__(self, detector, image_paths, conf_threshold, iou_threshold):
        super().__init__()
        self.detector = detector
        self.image_paths = image_paths
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def run(self):
        """스레드 실행"""
        try:
            results = self.detector.detect_batch(
                self.image_paths,
                self.conf_threshold,
                self.iou_threshold,
                progress_callback=self.progress.emit
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """메인 윈도우 클래스"""
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.current_results = None
        self.current_images = []
        self.visualizer = WaferMapVisualizer()
        self.report_gen = ReportGenerator()
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Solder Bump Defect Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout(central_widget)
        
        # 왼쪽 패널 (설정)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 오른쪽 패널 (결과)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_left_panel(self) -> QWidget:
        """왼쪽 설정 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 제목
        title = QLabel("Defect Detection System")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #2C3E50;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 검사 유형 선택
        inspection_group = QGroupBox("Inspection Type")
        inspection_layout = QVBoxLayout()
        
        self.inspection_combo = QComboBox()
        self.inspection_combo.addItems(["Microscope", "X-ray"])
        self.inspection_combo.currentTextChanged.connect(self.on_inspection_type_changed)
        inspection_layout.addWidget(self.inspection_combo)
        
        inspection_group.setLayout(inspection_layout)
        layout.addWidget(inspection_group)
        
        # 모델 로드
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        
        self.model_label = QLabel("No model loaded")
        self.model_label.setWordWrap(True)
        model_layout.addWidget(self.model_label)
        
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 이미지 입력
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        
        single_img_btn = QPushButton("Select Single Image")
        single_img_btn.clicked.connect(self.select_single_image)
        input_layout.addWidget(single_img_btn)
        
        folder_btn = QPushButton("Select Folder (Batch)")
        folder_btn.clicked.connect(self.select_folder)
        input_layout.addWidget(folder_btn)
        
        self.image_count_label = QLabel("No images selected")
        input_layout.addWidget(self.image_count_label)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 검출 설정
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        # 신뢰도 임계값
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.0, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.25)
        conf_layout.addWidget(self.conf_spinbox)
        settings_layout.addLayout(conf_layout)
        
        # IOU 임계값
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IOU:"))
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45)
        iou_layout.addWidget(self.iou_spinbox)
        settings_layout.addLayout(iou_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 검출 실행
        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.setStyleSheet("background-color: #3498DB; color: white; font-size: 14px; padding: 10px;")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        layout.addWidget(self.detect_btn)
        
        # 진행률
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 리포트 생성
        report_group = QGroupBox("Export")
        report_layout = QVBoxLayout()
        
        pdf_btn = QPushButton("Generate PDF Report")
        pdf_btn.clicked.connect(self.generate_pdf_report)
        report_layout.addWidget(pdf_btn)
        
        excel_btn = QPushButton("Generate Excel Report")
        excel_btn.clicked.connect(self.generate_excel_report)
        report_layout.addWidget(excel_btn)
        
        latex_btn = QPushButton("Generate LaTeX Table")
        latex_btn.clicked.connect(self.generate_latex_table)
        report_layout.addWidget(latex_btn)
        
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """오른쪽 결과 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 탭 위젯
        self.tab_widget = QTabWidget()
        
        # 로그 탭
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.tab_widget.addTab(self.log_text, "Log")
        
        # 결과 탭
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        self.results_label = QLabel("No results yet")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_label.setStyleSheet("font-size: 14px; color: #7F8C8D;")
        results_layout.addWidget(self.results_label)
        
        self.tab_widget.addTab(results_widget, "Results")
        
        # 시각화 탭
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        self.viz_label = QLabel("Run detection to see visualizations")
        self.viz_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viz_layout.addWidget(self.viz_label)
        
        self.tab_widget.addTab(viz_widget, "Visualization")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def log(self, message: str):
        """로그 메시지 추가"""
        self.log_text.append(message)
    
    def on_inspection_type_changed(self, text: str):
        """검사 유형 변경 핸들러"""
        inspection_type = text.lower()
        self.log(f"검사 유형 변경: {text}")
        
        if self.detector:
            self.detector = DefectDetector(inspection_type)
            self.model_label.setText("Model unloaded (type changed)")
            self.detect_btn.setEnabled(False)
    
    def load_model(self):
        """모델 로드"""
        inspection_type = self.inspection_combo.currentText().lower()
        
        # 기본 모델 경로
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            f"{inspection_type}_best.pt"
        )
        
        # 파일 선택
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            default_path if os.path.exists(default_path) else "",
            "PyTorch Model (*.pt)"
        )
        
        if file_path:
            self.detector = DefectDetector(inspection_type)
            if self.detector.load_model(file_path):
                self.model_label.setText(f"✓ Loaded: {os.path.basename(file_path)}")
                self.log(f"✓ 모델 로드 성공: {file_path}")
                self.update_detect_button_state()
            else:
                self.model_label.setText("✗ Failed to load model")
                self.log(f"✗ 모델 로드 실패: {file_path}")
    
    def select_single_image(self):
        """단일 이미지 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_images = [file_path]
            self.image_count_label.setText(f"1 image selected")
            self.log(f"이미지 선택: {file_path}")
            self.update_detect_button_state()
    
    def select_folder(self):
        """폴더 선택 (일괄 처리)"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder"
        )
        
        if folder_path:
            # 이미지 파일 찾기
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
            self.current_images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(image_extensions)
            ]
            
            count = len(self.current_images)
            self.image_count_label.setText(f"{count} images selected")
            self.log(f"폴더 선택: {folder_path} ({count}개 이미지)")
            self.update_detect_button_state()
    
    def update_detect_button_state(self):
        """검출 버튼 활성화 상태 업데이트"""
        enabled = (
            self.detector is not None and
            self.detector.model is not None and
            len(self.current_images) > 0
        )
        self.detect_btn.setEnabled(enabled)
    
    def run_detection(self):
        """검출 실행"""
        if not self.detector or not self.current_images:
            return
        
        self.log(f"\n{'='*60}")
        self.log(f"검출 시작: {len(self.current_images)}개 이미지")
        self.log(f"{'='*60}")
        
        # UI 비활성화
        self.detect_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 검출 스레드 시작
        self.detection_thread = DetectionThread(
            self.detector,
            self.current_images,
            self.conf_spinbox.value(),
            self.iou_spinbox.value()
        )
        self.detection_thread.progress.connect(self.on_detection_progress)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.start()
    
    def on_detection_progress(self, current: int, total: int):
        """검출 진행률 업데이트"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.log(f"진행: {current}/{total}")
    
    def on_detection_finished(self, results: List):
        """검출 완료 핸들러"""
        self.current_results = results
        
        # 통계 계산
        statistics = self.detector.get_statistics(results)
        
        # 로그 출력
        self.log(f"\n{'='*60}")
        self.log("검출 완료!")
        self.log(f"{'='*60}")
        self.log(f"총 이미지: {statistics['total_images']}")
        self.log(f"총 검출: {statistics['total_detections']}")
        self.log(f"평균 신뢰도: {statistics['avg_confidence']:.3f}")
        self.log(f"평균 추론 시간: {statistics['avg_inference_time']:.4f}s")
        self.log(f"FPS: {statistics['fps']:.2f}")
        self.log(f"\n클래스별 검출:")
        for class_name, count in statistics['class_counts'].items():
            self.log(f"  - {class_name}: {count}")
        
        # 결과 표시
        self.display_results(results, statistics)
        
        # UI 활성화
        self.detect_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        QMessageBox.information(self, "완료", "검출이 완료되었습니다!")
    
    def on_detection_error(self, error_msg: str):
        """검출 오류 핸들러"""
        self.log(f"\n✗ 오류 발생: {error_msg}")
        self.detect_btn.setEnabled(True)
        QMessageBox.critical(self, "오류", f"검출 중 오류가 발생했습니다:\n{error_msg}")
    
    def display_results(self, results: List, statistics: Dict):
        """결과 표시"""
        # 결과 탭 업데이트
        results_text = f"""
<h2>Detection Results</h2>
<table border="1" cellpadding="5">
<tr><td><b>Total Images</b></td><td>{statistics['total_images']}</td></tr>
<tr><td><b>Total Detections</b></td><td>{statistics['total_detections']}</td></tr>
<tr><td><b>Avg Detections/Image</b></td><td>{statistics['avg_detections_per_image']:.2f}</td></tr>
<tr><td><b>Avg Confidence</b></td><td>{statistics['avg_confidence']:.3f}</td></tr>
<tr><td><b>Avg Inference Time</b></td><td>{statistics['avg_inference_time']:.4f} s</td></tr>
<tr><td><b>FPS</b></td><td>{statistics['fps']:.2f}</td></tr>
</table>

<h3>Class Distribution</h3>
<table border="1" cellpadding="5">
<tr><th>Class</th><th>Count</th></tr>
"""
        for class_name, count in statistics['class_counts'].items():
            results_text += f"<tr><td>{class_name}</td><td>{count}</td></tr>"
        results_text += "</table>"
        
        self.results_label.setText(results_text)
        self.results_label.setTextFormat(Qt.TextFormat.RichText)
        
        # TODO: 시각화 탭 업데이트
    
    def generate_pdf_report(self):
        """PDF 리포트 생성"""
        if not self.current_results:
            QMessageBox.warning(self, "경고", "먼저 검출을 실행하세요.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            "defect_report.pdf",
            "PDF Files (*.pdf)"
        )
        
        if file_path:
            statistics = self.detector.get_statistics(self.current_results)
            if self.report_gen.generate_pdf_report(self.current_results, statistics, file_path):
                QMessageBox.information(self, "완료", f"PDF 리포트가 생성되었습니다:\n{file_path}")
    
    def generate_excel_report(self):
        """Excel 리포트 생성"""
        if not self.current_results:
            QMessageBox.warning(self, "경고", "먼저 검출을 실행하세요.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Excel Report",
            "defect_report.xlsx",
            "Excel Files (*.xlsx)"
        )
        
        if file_path:
            statistics = self.detector.get_statistics(self.current_results)
            if self.report_gen.generate_excel_report(self.current_results, statistics, file_path):
                QMessageBox.information(self, "완료", f"Excel 리포트가 생성되었습니다:\n{file_path}")
    
    def generate_latex_table(self):
        """LaTeX 테이블 생성"""
        if not self.current_results:
            QMessageBox.warning(self, "경고", "먼저 검출을 실행하세요.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LaTeX Table",
            "defect_table.tex",
            "LaTeX Files (*.tex)"
        )
        
        if file_path:
            statistics = self.detector.get_statistics(self.current_results)
            if self.report_gen.generate_latex_table(statistics, file_path):
                QMessageBox.information(self, "완료", f"LaTeX 테이블이 생성되었습니다:\n{file_path}")
