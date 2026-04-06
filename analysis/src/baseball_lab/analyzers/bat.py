import math
from ultralytics import YOLO
from typing import Dict, Any

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
        
        # 速度計測用の状態
        self._bat_history = []
        self._max_bat_speed = 0.0
        self._metrics_history = {}

    @property
    def name(self) -> str:
        return "bat"

    def _smooth_metric(self, name: str, value: float, weight: float) -> float:
        if name not in self._metrics_history:
            self._metrics_history[name] = value
            return value
        self._metrics_history[name] = (
            self._metrics_history[name] * (1.0 - weight) + value * weight
        )
        return self._metrics_history[name]

    def analyze_frame(
        self, frame, frame_idx: int, fps: float, context: dict = None
    ) -> Dict[str, Any]:
        # 純粋にYOLOで推論
        results = self.model.predict(
            frame, conf=self.conf_threshold, classes=[self.BAT_CLASS_ID], verbose=False
        )

        detected_boxes = []
        best_box = None
        max_conf = 0.0

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # [x1, y1, x2, y2, conf, cls]
                b = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                detected_boxes.append({"bbox": b, "confidence": conf})
                if conf > max_conf:
                    max_conf = conf
                    best_box = b

        metrics = {}
        if best_box and context:
            pose_metrics = context.get("pose", {}).get("metrics", {})
            m_per_norm = pose_metrics.get("m_per_norm")
            grip_pos = pose_metrics.get("grip_position")
            
            if m_per_norm is not None and grip_pos is not None:
                frame_height, frame_width = frame.shape[:2]
                
                # m_per_normは正規化座標（縦の長さ=1.0）に対するメートル
                m_per_pixel = m_per_norm / frame_height
                
                # グリップ位置をピクセル座標に変換
                grip_px = {
                    "x": grip_pos["x"] * frame_width,
                    "y": grip_pos["y"] * frame_height
                }
                
                # バウンディングボックスの4頂点から、グリップから最も遠い点をヘッドと推定
                x1, y1, x2, y2 = best_box
                corners = [
                    {"x": x1, "y": y1},
                    {"x": x2, "y": y1},
                    {"x": x1, "y": y2},
                    {"x": x2, "y": y2}
                ]
                
                head_pos = max(
                    corners, 
                    key=lambda p: math.hypot(p["x"] - grip_px["x"], p["y"] - grip_px["y"])
                )
                
                # 速度計算
                timestamp = frame_idx / fps
                self._bat_history.append({"pos": head_pos, "time": timestamp})
                
                if len(self._bat_history) > 2:
                    self._bat_history.pop(0)
                    
                if len(self._bat_history) == 2:
                    dt = self._bat_history[1]["time"] - self._bat_history[0]["time"]
                    if dt > 0:
                        prev_pos = self._bat_history[0]["pos"]
                        dist_pixel = math.hypot(head_pos["x"] - prev_pos["x"], head_pos["y"] - prev_pos["y"])
                        dist_m = dist_pixel * m_per_pixel
                        raw_speed_kmh = (dist_m / dt) * 3.6
                        
                        # 異常値（ノイズ等による瞬間移動）を除外
                        max_speed_threshold = 250.0
                        if raw_speed_kmh < max_speed_threshold:
                            # スムージング適用 (瞬間最大速度を捉えるためweight=1.0とする)
                            speed_kmh = self._smooth_metric("bat_speed", raw_speed_kmh, weight=1.0)
                        else:
                            speed_kmh = self._metrics_history.get("bat_speed", 0.0)
                        
                        self._max_bat_speed = max(self._max_bat_speed, speed_kmh)
                        metrics["bat_speed"] = speed_kmh
                        
        metrics["max_bat_speed"] = self._max_bat_speed

        return {"detections": detected_boxes, "metrics": metrics}

    def reset(self):
        self._bat_history = []
        self._max_bat_speed = 0.0
        self._metrics_history = {}

    def close(self):
        pass
