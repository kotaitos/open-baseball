import math
import os
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

from baseball_lab.analyzers.base import BaseAnalyzer
from baseball_lab.core.filter import OneEuroFilter
from baseball_lab.core.metrics import calculate_distance_3d, calculate_rotation


class PoseAnalyzer(BaseAnalyzer):
    def __init__(
        self, model_path: str = None, player_height_m: float = 1.93, config: dict = None
    ):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "pose_landmarker_heavy.task"
            )

        # 設定のデフォルトフォールバック
        self.config = config or {}
        pose_config = self.config.get("pose", {})

        self.BODY_HEIGHT_TO_STEM_RATIO = pose_config.get(
            "body_height_to_stem_ratio", 0.85
        )
        self.MIN_LANDMARK_VISIBILITY = pose_config.get("min_landmark_visibility", 0.15)
        self.MIN_GRIP_VISIBILITY = pose_config.get("min_grip_visibility", 0.5)
        self.GRIP_VELOCITY_DECAY = pose_config.get("grip_velocity_decay", 0.99)
        self._max_gap_frames = pose_config.get("max_gap_frames", 20)

        mp_config = pose_config.get("mediapipe", {})
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=mp_config.get(
                "min_pose_detection_confidence", 0.5
            ),
            min_pose_presence_confidence=mp_config.get(
                "min_pose_presence_confidence", 0.5
            ),
            min_tracking_confidence=mp_config.get("min_tracking_confidence", 0.6),
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        self.player_height_m = player_height_m
        self._max_body_norm = 0.0
        self._max_grip_speed = 0.0
        self._max_separation = 0.0

        # フィルタリング状態
        self._grip_filters = None
        self._pos_history = {}
        self._raw_grip_history = []
        self._metrics_history = {}

        # 欠損補完用
        self._last_valid_grip = None
        self._last_grip_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._gap_count = 0

    @property
    def name(self) -> str:
        return "pose"

    def _get_raw_landmark(
        self, landmarks: List[Dict[str, Any]], lm_id: int
    ) -> Optional[Dict[str, float]]:
        lm = next((lm for lm in landmarks if lm["id"] == lm_id), None)
        if not lm or (lm["visibility"] < self.MIN_LANDMARK_VISIBILITY):
            return None
        return {
            "x": lm["x"],
            "y": lm["y"],
            "z": lm["z"],
            "visibility": lm["visibility"],
        }

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
        if context is None:
            context = {}

        pose_config = self.config.get("pose", {})
        filter_config = pose_config.get("filter", {})
        smooth_config = pose_config.get("smoothing", {})

        # フィルタの遅延初期化（実際の解析FPSに合わせる）
        if self._grip_filters is None:
            mc = filter_config.get("mincutoff", 0.5)
            b = filter_config.get("beta", 0.1)
            self._grip_filters = {
                "x": OneEuroFilter(fps, mincutoff=mc, beta=b),
                "y": OneEuroFilter(fps, mincutoff=mc, beta=b),
                "z": OneEuroFilter(fps, mincutoff=mc, beta=b),
            }

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(1000 * frame_idx / fps)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        raw_lms = []
        if detection_result.pose_landmarks:
            for landmarks in detection_result.pose_landmarks:
                for idx, lm in enumerate(landmarks):
                    raw_lms.append(
                        {
                            "id": idx,
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility,
                            "presence": lm.presence,
                        }
                    )

        # 1. スケーリング算出
        if raw_lms:
            l_sh, r_sh = (
                self._get_raw_landmark(raw_lms, 11),
                self._get_raw_landmark(raw_lms, 12),
            )
            l_ank, r_ank = (
                self._get_raw_landmark(raw_lms, 27),
                self._get_raw_landmark(raw_lms, 28),
            )
            if l_sh and r_sh and l_ank and r_ank:
                current_body_norm = abs(
                    ((l_sh["y"] + r_sh["y"]) / 2) - ((l_ank["y"] + r_ank["y"]) / 2)
                )
                if current_body_norm > self._max_body_norm:
                    self._max_body_norm = current_body_norm

        raw_m_per_norm = (self.player_height_m * self.BODY_HEIGHT_TO_STEM_RATIO) / max(
            self._max_body_norm, 0.01
        )
        m_per_norm = self._smooth_metric("m_per_norm", raw_m_per_norm, weight=0.05)

        # 2. グリップ位置（拳の重心）の算出
        curr_grip_raw = None
        if raw_lms:
            # 親指(21, 22)を追加
            l_fist_ids = [15, 17, 19, 21]
            r_fist_ids = [16, 18, 20, 22]

            def get_fist_center(ids):
                lms = [self._get_raw_landmark(raw_lms, i) for i in ids]
                # 視認性が極端に低いノイズを除外
                lms = [lm for lm in lms if lm is not None and lm.get("visibility", 0) >= 0.1]
                if not lms:
                    return None

                total_v = sum(lm["visibility"] for lm in lms)
                if total_v < 0.1:
                    return None

                return {
                    "x": sum(lm["x"] * lm["visibility"] for lm in lms) / total_v,
                    "y": sum(lm["y"] * lm["visibility"] for lm in lms) / total_v,
                    "z": sum(lm["z"] * lm["visibility"] for lm in lms) / total_v,
                    "visibility": total_v / len(ids),  # 全体に対する信頼度
                }

            l_fist = get_fist_center(l_fist_ids)
            r_fist = get_fist_center(r_fist_ids)

            if l_fist and r_fist:
                # 異常距離のチェック (手同士が離れすぎている場合は誤認識とみなす)
                dist = math.hypot(l_fist["x"] - r_fist["x"], l_fist["y"] - r_fist["y"])
                
                lv = l_fist["visibility"]
                rv = r_fist["visibility"]

                if dist > 0.2:
                    # 離れすぎている場合は視認性の高い方を採用
                    if lv > rv:
                        curr_grip_raw = l_fist
                    else:
                        curr_grip_raw = r_fist
                elif lv > 0.8 and rv < 0.3:
                    # 片手が極端に隠れている場合
                    curr_grip_raw = l_fist
                elif rv > 0.8 and lv < 0.3:
                    curr_grip_raw = r_fist
                else:
                    lv_weight = max(lv, 0.01)
                    rv_weight = max(rv, 0.01)
                    sum_v = lv_weight + rv_weight
                    curr_grip_raw = {
                        k: (l_fist[k] * lv_weight + r_fist[k] * rv_weight) / sum_v
                        for k in ["x", "y", "z"]
                    }
            elif l_fist:
                curr_grip_raw = l_fist
            elif r_fist:
                curr_grip_raw = r_fist

        # 3. フィルタリング (One Euro Filter)
        if curr_grip_raw:
            raw_x, raw_y = curr_grip_raw["x"], curr_grip_raw["y"]
            curr_grip_raw = {
                k: self._grip_filters[k].filter(curr_grip_raw[k])
                for k in ["x", "y", "z"]
            }

            hard_reset_threshold = filter_config.get("hard_reset_threshold", 0.1)
            dist_error = math.hypot(
                curr_grip_raw["x"] - raw_x, curr_grip_raw["y"] - raw_y
            )
            if dist_error > hard_reset_threshold:
                curr_grip_raw = {"x": raw_x, "y": raw_y, "z": curr_grip_raw["z"]}
                for k in ["x", "y", "z"]:
                    self._grip_filters[k].x_prev = curr_grip_raw[k]

            if self._last_valid_grip:
                self._last_grip_velocity = {
                    k: (curr_grip_raw[k] - self._last_valid_grip[k])
                    for k in ["x", "y", "z"]
                }
            self._last_valid_grip = curr_grip_raw
            self._gap_count = 0
        elif self._last_valid_grip and self._gap_count < self._max_gap_frames:
            self._gap_count += 1
            self._last_grip_velocity = {
                k: v * self.GRIP_VELOCITY_DECAY
                for k, v in self._last_grip_velocity.items()
            }
            curr_grip_raw = {
                k: self._last_valid_grip[k] + self._last_grip_velocity[k]
                for k in ["x", "y", "z"]
            }
            self._last_valid_grip = curr_grip_raw

        # 4. メトリクス更新
        smoothed_metrics = {"m_per_norm": m_per_norm}
        if curr_grip_raw:
            smoothed_metrics["grip_position"] = curr_grip_raw
            self._raw_grip_history.append(
                {"pos": curr_grip_raw, "time": timestamp_ms / 1000.0}
            )
            if len(self._raw_grip_history) > 2:
                self._raw_grip_history.pop(0)
            if len(self._raw_grip_history) == 2:
                dt = (
                    self._raw_grip_history[1]["time"]
                    - self._raw_grip_history[0]["time"]
                )
                if dt > 0:
                    dist_m = (
                        calculate_distance_3d(
                            self._raw_grip_history[0]["pos"],
                            self._raw_grip_history[1]["pos"],
                        )
                        * m_per_norm
                    )
                    raw_speed = (dist_m / dt) * 3.6
                    max_speed = pose_config.get("max_grip_speed_kmh", 160.0)
                    speed_weight = smooth_config.get("grip_speed_weight", 0.8)

                    if raw_speed < max_speed:
                        smoothed_metrics["grip_speed"] = self._smooth_metric(
                            "grip_speed", raw_speed, weight=speed_weight
                        )
                    else:
                        smoothed_metrics["grip_speed"] = self._metrics_history.get(
                            "grip_speed", 0.0
                        )

        # 捻転差
        if raw_lms:
            l_hip, r_hip = (
                self._get_raw_landmark(raw_lms, 23),
                self._get_raw_landmark(raw_lms, 24),
            )
            l_sh, r_sh = (
                self._get_raw_landmark(raw_lms, 11),
                self._get_raw_landmark(raw_lms, 12),
            )
            if l_hip and r_hip and l_sh and r_sh:
                sep = abs(
                    calculate_rotation(l_sh, r_sh) - calculate_rotation(l_hip, r_hip)
                )
                sep_weight = smooth_config.get("separation_weight", 0.5)
                smoothed_metrics["separation"] = self._smooth_metric(
                    "separation", sep, weight=sep_weight
                )

        # 最大値更新
        self._max_grip_speed = max(
            self._max_grip_speed, smoothed_metrics.get("grip_speed", 0.0)
        )
        self._max_separation = max(
            self._max_separation, smoothed_metrics.get("separation", 0.0)
        )
        smoothed_metrics.update(
            {
                "max_grip_speed": self._max_grip_speed,
                "max_separation": self._max_separation,
            }
        )

        return {"landmarks": raw_lms, "metrics": smoothed_metrics}

    def reset(self):
        self._max_grip_speed = 0.0
        self._max_body_norm = 0.0
        self._max_separation = 0.0
        self._last_valid_grip = None
        self._gap_count = 0
        self._metrics_history = {}
        self._raw_grip_history = []
        if self._grip_filters:
            for f in self._grip_filters.values():
                f.x_prev = None

    def close(self):
        self.landmarker.close()
