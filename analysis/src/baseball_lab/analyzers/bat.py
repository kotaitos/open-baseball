import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, List

from baseball_lab.analyzers.base import BaseAnalyzer

class BatAnalyzer(BaseAnalyzer):
    def __init__(self, config: dict = None):
        self.config = config or {}
        bat_config = self.config.get("bat", {})
        
        # モデル名は設定から取得（デフォルト yolo12s.pt）
        model_name = bat_config.get("yolo_model", "yolo12s.pt")
        self.model = YOLO(model_name)
        self.conf_threshold = bat_config.get("confidence_threshold", 0.1)
        
        # クラスID 34 が 'baseball bat'
        self.BAT_CLASS_ID = 34

    @property
    def name(self) -> str:
        return "bat"

    def analyze_frame(self, frame, frame_idx: int, fps: float, context: dict = None) -> Dict[str, Any]:
        # 純粋にYOLOで推論
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            classes=[self.BAT_CLASS_ID], 
            verbose=False
        )
        
        detected_boxes = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # [x1, y1, x2, y2, conf, cls]
                b = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                detected_boxes.append({
                    "bbox": b,
                    "confidence": conf
                })
        
        return {
            "detections": detected_boxes
        }

    def reset(self):
        pass

    def close(self):
        pass
