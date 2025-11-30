"""
리포트 생성 유틸리티
"""

import os
from typing import List, Dict
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt


class ReportGenerator:
    """리포트 생성기 클래스"""
    
    def __init__(self):
        """초기화"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """커스텀 스타일 설정"""
        # 제목 스타일
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # 소제목 스타일
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            spaceBefore=12
        ))
    
    def generate_pdf_report(
        self,
        results: List[Dict],
        statistics: Dict,
        output_path: str,
        figures: Dict[str, str] = None
    ) -> bool:
        """
        PDF 리포트 생성
        
        Args:
            results: 검출 결과 리스트
            statistics: 통계 정보
            output_path: 출력 파일 경로
            figures: 그래프 이미지 경로 딕셔너리
        
        Returns:
            성공 여부
        """
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # 제목
            title = Paragraph("Solder Bump Defect Detection Report", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 0.2 * inch))
            
            # 날짜
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            date_para = Paragraph(f"<b>Report Date:</b> {date_str}", self.styles['Normal'])
            story.append(date_para)
            story.append(Spacer(1, 0.3 * inch))
            
            # 요약 통계
            story.append(Paragraph("Summary Statistics", self.styles['CustomHeading']))
            summary_data = [
                ["Metric", "Value"],
                ["Total Images", str(statistics['total_images'])],
                ["Total Detections", str(statistics['total_detections'])],
                ["Avg Detections/Image", f"{statistics['avg_detections_per_image']:.2f}"],
                ["Avg Confidence", f"{statistics['avg_confidence']:.3f}"],
                ["Avg Inference Time", f"{statistics['avg_inference_time']:.4f} s"],
                ["FPS", f"{statistics['fps']:.2f}"],
            ]
            
            summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.3 * inch))
            
            # 클래스별 검출 수
            story.append(Paragraph("Class Distribution", self.styles['CustomHeading']))
            class_data = [["Class", "Count"]]
            for class_name, count in statistics['class_counts'].items():
                class_data.append([class_name, str(count)])
            
            class_table = Table(class_data, colWidths=[3 * inch, 2 * inch])
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ECC71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(class_table)
            
            # 그래프 추가
            if figures:
                story.append(PageBreak())
                story.append(Paragraph("Visualization", self.styles['CustomHeading']))
                
                for fig_name, fig_path in figures.items():
                    if os.path.exists(fig_path):
                        img = Image(fig_path, width=5 * inch, height=4 * inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2 * inch))
            
            # PDF 생성
            doc.build(story)
            print(f"✓ PDF 리포트 생성 완료: {output_path}")
            return True
        
        except Exception as e:
            print(f"✗ PDF 리포트 생성 실패: {e}")
            return False
    
    def generate_excel_report(
        self,
        results: List[Dict],
        statistics: Dict,
        output_path: str
    ) -> bool:
        """
        Excel 리포트 생성
        
        Args:
            results: 검출 결과 리스트
            statistics: 통계 정보
            output_path: 출력 파일 경로
        
        Returns:
            성공 여부
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 요약 시트
                summary_df = pd.DataFrame({
                    'Metric': [
                        'Total Images',
                        'Total Detections',
                        'Avg Detections/Image',
                        'Avg Confidence',
                        'Avg Inference Time (s)',
                        'FPS'
                    ],
                    'Value': [
                        statistics['total_images'],
                        statistics['total_detections'],
                        f"{statistics['avg_detections_per_image']:.2f}",
                        f"{statistics['avg_confidence']:.3f}",
                        f"{statistics['avg_inference_time']:.4f}",
                        f"{statistics['fps']:.2f}"
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 클래스 분포 시트
                class_df = pd.DataFrame({
                    'Class': list(statistics['class_counts'].keys()),
                    'Count': list(statistics['class_counts'].values())
                })
                class_df.to_excel(writer, sheet_name='Class Distribution', index=False)
                
                # 상세 검출 결과 시트
                detail_rows = []
                for result in results:
                    if 'detections' in result:
                        for det in result['detections']:
                            detail_rows.append({
                                'Image': os.path.basename(result['image_path']),
                                'Class': det['class_name'],
                                'Confidence': f"{det['confidence']:.3f}",
                                'X': f"{det['bbox']['center_x']:.1f}",
                                'Y': f"{det['bbox']['center_y']:.1f}",
                                'Width': f"{det['bbox']['width']:.1f}",
                                'Height': f"{det['bbox']['height']:.1f}"
                            })
                
                if detail_rows:
                    detail_df = pd.DataFrame(detail_rows)
                    detail_df.to_excel(writer, sheet_name='Detections', index=False)
            
            print(f"✓ Excel 리포트 생성 완료: {output_path}")
            return True
        
        except Exception as e:
            print(f"✗ Excel 리포트 생성 실패: {e}")
            return False
    
    def generate_latex_table(
        self,
        statistics: Dict,
        output_path: str
    ) -> bool:
        """
        LaTeX 테이블 생성 (논문용)
        
        Args:
            statistics: 통계 정보
            output_path: 출력 파일 경로
        
        Returns:
            성공 여부
        """
        try:
            latex_content = r"""\begin{table}[h]
\centering
\caption{Defect Detection Results}
\label{tab:detection_results}
\begin{tabular}{|l|r|}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Total Images & """ + str(statistics['total_images']) + r""" \\
Total Detections & """ + str(statistics['total_detections']) + r""" \\
Avg Detections/Image & """ + f"{statistics['avg_detections_per_image']:.2f}" + r""" \\
Avg Confidence & """ + f"{statistics['avg_confidence']:.3f}" + r""" \\
Avg Inference Time (s) & """ + f"{statistics['avg_inference_time']:.4f}" + r""" \\
FPS & """ + f"{statistics['fps']:.2f}" + r""" \\
\hline
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Defect Class Distribution}
\label{tab:class_distribution}
\begin{tabular}{|l|r|}
\hline
\textbf{Class} & \textbf{Count} \\
\hline
"""
            
            for class_name, count in statistics['class_counts'].items():
                latex_content += f"{class_name} & {count} \\\\\n"
            
            latex_content += r"""\hline
\end{tabular}
\end{table}
"""
            
            with open(output_path, 'w') as f:
                f.write(latex_content)
            
            print(f"✓ LaTeX 테이블 생성 완료: {output_path}")
            return True
        
        except Exception as e:
            print(f"✗ LaTeX 테이블 생성 실패: {e}")
            return False
