"""
YOLOv8 기반 솔더 범프 결함 검출기
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time


class DefectDetector:
    """결함 검출기 클래스"""
    
    # 검사 유형별 클래스 정의
    MICROSCOPE_CLASSES = {
        0: "Type1",
        1: "Type2",
        2: "Type3",
        3: "Type4"
    }
    
    XRAY_CLASSES = {
        0: "Void",
        1: "Bridge",
        2: "HiP",
        3: "ColdJoint",
        4: "Crack",
        5: "Normal"
    }
    
    def __init__(self, inspection_type: str = "microscope"):
        """
        초기화
        
        Args:
            inspection_type: 검사 유형 ("microscope" 또는 "xray")
        """
        self.inspection_type = inspection_type
        self.model = None
        self.model_path = None
        self.classes = (
            self.MICROSCOPE_CLASSES if inspection_type == "microscope"
            else self.XRAY_CLASSES
        )
    
    def load_model(self, model_path: str) -> bool:
        """
        모델 로드
        
        Args:
            model_path: 모델 파일 경로 (.pt)
        
        Returns:
            성공 여부
        """
        try:
            if not os.path.exists(model_path):
                print(f"모델 파일을 찾을 수 없습니다: {model_path}")
                return False
            
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"✓ 모델 로드 완료: {model_path}")
            return True
        
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
            return False
    
    def detect(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict:
        """
        단일 이미지 결함 검출
        
        Args:
            image_path: 이미지 파일 경로
            conf_threshold: 신뢰도 임계값
            iou_threshold: IOU 임계값
        
        Returns:
            검출 결과 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 시작 시간
        start_time = time.time()
        
        # 추론
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )[0]
        
        # 추론 시간
        inference_time = time.time() - start_time
        
        # 결과 파싱
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            
            # 바운딩 박스 좌표 (xyxy 형식)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 클래스 및 신뢰도
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_name = self.classes.get(class_id, f"Unknown_{class_id}")
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2)
                }
            })
        
        # 이미지 정보
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        return {
            "image_path": image_path,
            "image_size": {"width": img_width, "height": img_height},
            "detections": detections,
            "num_detections": len(detections),
            "inference_time": inference_time,
            "model_path": self.model_path,
            "inspection_type": self.inspection_type
        }
    
    def detect_batch(
        self,
        image_paths: List[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        progress_callback=None
    ) -> List[Dict]:
        """
        여러 이미지 일괄 검출
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            conf_threshold: 신뢰도 임계값
            iou_threshold: IOU 임계값
            progress_callback: 진행률 콜백 함수 (current, total)
        
        Returns:
            검출 결과 리스트
        """
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect(image_path, conf_threshold, iou_threshold)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total)
            
            except Exception as e:
                print(f"✗ 검출 실패 ({image_path}): {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "num_detections": 0
                })
        
        return results
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        검출 결과 통계 계산
        
        Args:
            results: detect_batch() 결과
        
        Returns:
            통계 딕셔너리
        """
        total_images = len(results)
        total_detections = sum(r.get("num_detections", 0) for r in results)
        
        # 클래스별 검출 수
        class_counts = {name: 0 for name in self.classes.values()}
        confidences = []
        inference_times = []
        
        for result in results:
            if "detections" in result:
                for det in result["detections"]:
                    class_name = det["class_name"]
                    class_counts[class_name] += 1
                    confidences.append(det["confidence"])
            
            if "inference_time" in result:
                inference_times.append(result["inference_time"])
        
        # 평균 계산
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        return {
            "total_images": total_images,
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / total_images if total_images > 0 else 0.0,
            "class_counts": class_counts,
            "avg_confidence": avg_confidence,
            "avg_inference_time": avg_inference_time,
            "fps": fps
        }
    
    def get_model_info(self) -> Dict:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_path": self.model_path,
            "inspection_type": self.inspection_type,
            "classes": self.classes,
            "num_classes": len(self.classes)
        }
