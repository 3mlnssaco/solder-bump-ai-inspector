"""
웨이퍼맵 시각화 유틸리티
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from typing import List, Dict, Tuple
import seaborn as sns


class WaferMapVisualizer:
    """웨이퍼맵 시각화 클래스"""
    
    def __init__(self, wafer_size: Tuple[int, int] = (300, 300)):
        """
        초기화
        
        Args:
            wafer_size: 웨이퍼 크기 (mm 단위)
        """
        self.wafer_size = wafer_size
        self.dpi = 100
    
    def create_defect_map(
        self,
        detections: List[Dict],
        image_size: Tuple[int, int],
        title: str = "Defect Distribution Map"
    ) -> Figure:
        """
        결함 분포 맵 생성
        
        Args:
            detections: 검출 결과 리스트
            image_size: 이미지 크기 (width, height)
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)
        
        # 웨이퍼 원 그리기
        circle = plt.Circle((0.5, 0.5), 0.48, color='lightgray', fill=True, alpha=0.3)
        ax.add_patch(circle)
        
        # 결함 위치 표시
        if detections:
            img_width, img_height = image_size
            
            for det in detections:
                bbox = det["bbox"]
                # 정규화된 좌표 (0-1 범위)
                norm_x = bbox["center_x"] / img_width
                norm_y = bbox["center_y"] / img_height
                
                # 클래스별 색상
                color = self._get_class_color(det["class_name"])
                
                # 신뢰도에 따른 크기
                size = 100 + (det["confidence"] * 200)
                
                ax.scatter(
                    norm_x, norm_y,
                    s=size,
                    c=[color],
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=1,
                    label=det["class_name"]
                )
        
        # 축 설정
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X Position (normalized)', fontsize=12)
        ax.set_ylabel('Y Position (normalized)', fontsize=12)
        
        # 범례 (중복 제거)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_heatmap(
        self,
        detections: List[Dict],
        image_size: Tuple[int, int],
        grid_size: int = 20,
        title: str = "Defect Density Heatmap"
    ) -> Figure:
        """
        결함 밀도 히트맵 생성
        
        Args:
            detections: 검출 결과 리스트
            image_size: 이미지 크기
            grid_size: 그리드 크기
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        # 그리드 초기화
        heatmap = np.zeros((grid_size, grid_size))
        
        if detections:
            img_width, img_height = image_size
            
            for det in detections:
                bbox = det["bbox"]
                # 그리드 인덱스 계산
                grid_x = int((bbox["center_x"] / img_width) * grid_size)
                grid_y = int((bbox["center_y"] / img_height) * grid_size)
                
                # 범위 체크
                grid_x = min(grid_x, grid_size - 1)
                grid_y = min(grid_y, grid_size - 1)
                
                heatmap[grid_y, grid_x] += 1
        
        # 히트맵 그리기
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)
        
        sns.heatmap(
            heatmap,
            cmap='YlOrRd',
            annot=False,
            fmt='d',
            cbar_kws={'label': 'Defect Count'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X Grid', fontsize=12)
        ax.set_ylabel('Y Grid', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_class_distribution(
        self,
        class_counts: Dict[str, int],
        title: str = "Defect Class Distribution"
    ) -> Figure:
        """
        클래스별 검출 분포 막대 그래프
        
        Args:
            class_counts: 클래스별 검출 수
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = [self._get_class_color(c) for c in classes]
        
        bars = ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Defect Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_confidence_histogram(
        self,
        confidences: List[float],
        title: str = "Confidence Distribution"
    ) -> Figure:
        """
        신뢰도 분포 히스토그램
        
        Args:
            confidences: 신뢰도 리스트
            title: 그래프 제목
        
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        ax.hist(
            confidences,
            bins=20,
            color='skyblue',
            edgecolor='black',
            alpha=0.7
        )
        
        # 평균선
        if confidences:
            mean_conf = np.mean(confidences)
            ax.axvline(
                mean_conf,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Mean: {mean_conf:.3f}'
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _get_class_color(self, class_name: str) -> str:
        """
        클래스별 색상 반환
        
        Args:
            class_name: 클래스 이름
        
        Returns:
            색상 코드
        """
        color_map = {
            # 현미경
            "Type1": "#FF6B6B",
            "Type2": "#4ECDC4",
            "Type3": "#45B7D1",
            "Type4": "#FFA07A",
            
            # X-ray
            "Void": "#E74C3C",
            "Bridge": "#3498DB",
            "HiP": "#F39C12",
            "ColdJoint": "#9B59B6",
            "Crack": "#E67E22",
            "Normal": "#2ECC71"
        }
        
        return color_map.get(class_name, "#95A5A6")
