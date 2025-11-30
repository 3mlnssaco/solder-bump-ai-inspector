"""
논문용 평가 그래프 생성 유틸리티
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
import pandas as pd


class EvaluationVisualizer:
    """평가 시각화 클래스"""
    
    def __init__(self, dpi: int = 150):
        """
        초기화
        
        Args:
            dpi: 이미지 해상도 (논문용 고해상도)
        """
        self.dpi = dpi
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
    
    def create_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        title: str = "Confusion Matrix"
    ) -> Figure:
        """
        혼동 행렬 생성
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            class_names: 클래스 이름 리스트
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # 정규화된 혼동 행렬
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def create_precision_recall_curve(
        self,
        precisions: List[float],
        recalls: List[float],
        class_name: str = "All Classes",
        title: str = "Precision-Recall Curve"
    ) -> Figure:
        """
        정밀도-재현율 곡선 생성
        
        Args:
            precisions: 정밀도 리스트
            recalls: 재현율 리스트
            class_name: 클래스 이름
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        ax.plot(recalls, precisions, linewidth=2, label=class_name)
        ax.fill_between(recalls, precisions, alpha=0.2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_f1_confidence_curve(
        self,
        confidences: np.ndarray,
        f1_scores: np.ndarray,
        title: str = "F1 Score vs Confidence Threshold"
    ) -> Figure:
        """
        F1 점수-신뢰도 곡선 생성
        
        Args:
            confidences: 신뢰도 임계값 배열
            f1_scores: F1 점수 배열
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        ax.plot(confidences, f1_scores, linewidth=2, color='#2ECC71')
        ax.fill_between(confidences, f1_scores, alpha=0.2, color='#2ECC71')
        
        # 최대 F1 점수 표시
        max_idx = np.argmax(f1_scores)
        max_conf = confidences[max_idx]
        max_f1 = f1_scores[max_idx]
        
        ax.plot(max_conf, max_f1, 'ro', markersize=10, label=f'Max F1: {max_f1:.3f} @ {max_conf:.2f}')
        ax.axvline(max_conf, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Confidence Threshold', fontsize=14)
        ax.set_ylabel('F1 Score', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_training_curves(
        self,
        epochs: List[int],
        train_loss: List[float],
        val_loss: List[float],
        map50: List[float],
        title: str = "Training Curves"
    ) -> Figure:
        """
        학습 곡선 생성 (Loss + mAP)
        
        Args:
            epochs: 에폭 리스트
            train_loss: 학습 손실
            val_loss: 검증 손실
            map50: mAP@0.5
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Loss 곡선
        ax1.plot(epochs, train_loss, linewidth=2, label='Train Loss', color='#3498DB')
        ax1.plot(epochs, val_loss, linewidth=2, label='Val Loss', color='#E74C3C')
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # mAP 곡선
        ax2.plot(epochs, map50, linewidth=2, label='mAP@0.5', color='#2ECC71')
        ax2.fill_between(epochs, map50, alpha=0.2, color='#2ECC71')
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('mAP@0.5', fontsize=14)
        ax2.set_title('Mean Average Precision', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def create_detection_comparison(
        self,
        original_image_path: str,
        detected_image_path: str,
        title: str = "Detection Results"
    ) -> Figure:
        """
        원본 vs 검출 결과 비교
        
        Args:
            original_image_path: 원본 이미지 경로
            detected_image_path: 검출 결과 이미지 경로
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        from PIL import Image
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=self.dpi)
        
        # 원본 이미지
        original = Image.open(original_image_path)
        ax1.imshow(original)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 검출 결과
        detected = Image.open(detected_image_path)
        ax2.imshow(detected)
        ax2.set_title('Detection Result', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_performance_comparison(
        self,
        models: List[str],
        metrics: Dict[str, List[float]],
        title: str = "Model Performance Comparison"
    ) -> Figure:
        """
        모델 성능 비교 막대 그래프
        
        Args:
            models: 모델 이름 리스트
            metrics: 메트릭 딕셔너리 {'metric_name': [values]}
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        x = np.arange(len(models))
        width = 0.2
        multiplier = 0
        
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i % len(colors)])
            
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            multiplier += 1
        
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend(loc='upper left', fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_all_figures(
        self,
        figures: Dict[str, Figure],
        output_dir: str,
        format: str = 'png'
    ) -> Dict[str, str]:
        """
        모든 그래프를 파일로 저장
        
        Args:
            figures: 그래프 딕셔너리 {'name': Figure}
            output_dir: 출력 디렉토리
            format: 파일 형식 (png, pdf, svg)
        
        Returns:
            저장된 파일 경로 딕셔너리
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        for name, fig in figures.items():
            filename = f"{name}.{format}"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            saved_paths[name] = filepath
            print(f"✓ 저장: {filepath}")
            plt.close(fig)
        
        return saved_paths
