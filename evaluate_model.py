#!/usr/bin/env python3
"""
물리 기반 X-ray 검사 시스템 - 모델 평가 스크립트
Physics-based X-ray Inspection System - Model Evaluation Script
"""

from ultralytics import YOLO
import json
import os

def main():
    # Best model from physics_resume training
    model_path = 'runs/xray_detection/physics_resume/weights/best.pt'
    data_path = 'data/xray/data.yaml'

    print('='*60)
    print('물리 기반 X-ray 검사 시스템 - 모델 평가 결과')
    print('Physics-based X-ray Inspection System - Model Evaluation')
    print('='*60)

    # Load model
    model = YOLO(model_path)

    # Run validation
    results = model.val(
        data=data_path,
        batch=4,
        imgsz=640,
        device='cpu',
        verbose=True
    )

    print('\n' + '='*60)
    print('연구 논문용 평가 지표 (Research Paper Metrics)')
    print('='*60)

    # Extract metrics
    metrics = results.results_dict

    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    mAP50 = metrics.get('metrics/mAP50(B)', 0)
    mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)

    print('\n전체 성능 지표 (Overall Performance):')
    print(f'  - Precision (정밀도): {precision:.4f}')
    print(f'  - Recall (재현율): {recall:.4f}')
    print(f'  - mAP@50: {mAP50:.4f}')
    print(f'  - mAP@50-95: {mAP50_95:.4f}')

    # F1 Score calculation
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f'  - F1-Score: {f1:.4f}')

    # Per-class results
    print('\n클래스별 성능 (Per-Class AP@50):')
    class_names = ['Void', 'Bridge', 'HiP', 'ColdJoint', 'Crack', 'Normal']
    per_class_ap = {}

    if hasattr(results, 'ap50') and results.ap50 is not None:
        for i, name in enumerate(class_names):
            if i < len(results.ap50):
                ap = float(results.ap50[i])
                per_class_ap[name] = ap
                print(f'  - {name}: {ap:.4f}')

    # Loss values
    box_loss = metrics.get('val/box_loss', 0)
    cls_loss = metrics.get('val/cls_loss', 0)
    dfl_loss = metrics.get('val/dfl_loss', 0)

    print('\n손실 값 (Loss Values):')
    print(f'  - Box Loss: {box_loss:.4f}')
    print(f'  - Cls Loss: {cls_loss:.4f}')
    print(f'  - DFL Loss: {dfl_loss:.4f}')

    # Confusion matrix info
    print('\n모델 정보 (Model Information):')
    print(f'  - 모델: YOLOv8n')
    print(f'  - 학습 epochs: 15')
    print(f'  - 데이터셋: 물리 기반 시뮬레이션 X-ray')
    print(f'  - 이미지 수: 500장 (Train: 350, Val: 100, Test: 50)')
    print(f'  - 총 바운딩 박스: ~98,000개')
    print(f'  - 클래스 수: 6')

    # Save results to JSON
    results_summary = {
        'model': 'YOLOv8n (physics_resume)',
        'epochs': 15,
        'dataset': {
            'total_images': 500,
            'train_images': 350,
            'val_images': 100,
            'test_images': 50,
            'total_bboxes': 98000,
            'num_classes': 6,
            'classes': class_names
        },
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'mAP50': float(mAP50),
            'mAP50_95': float(mAP50_95),
            'f1_score': float(f1)
        },
        'per_class_ap50': per_class_ap,
        'loss': {
            'box_loss': float(box_loss),
            'cls_loss': float(cls_loss),
            'dfl_loss': float(dfl_loss)
        }
    }

    output_path = 'runs/xray_detection/physics_resume/evaluation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f'\n평가 결과가 {output_path}에 저장되었습니다.')
    print('='*60)

    return results_summary

if __name__ == '__main__':
    main()
