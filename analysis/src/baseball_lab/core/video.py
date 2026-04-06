import cv2
import json
import numpy as np

# MediaPipe ポーズコネクションの定義（骨格のつながり）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def visualize_pose(
    video_path: str,
    json_path: str,
    output_path: str,
    slow_mo_factor: int = 1,
    config: dict = None,
):
    """
    動画と解析結果のJSONを読み込み、骨格と各指標の最大値を可視化します。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    with open(json_path, "r") as f:
        analysis_data = json.load(f)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # スロー動画のフレームから解析データのインデックスを算出（同期）
        data_idx = frame_idx // slow_mo_factor

        if data_idx < len(analysis_data):
            frame_analysis = analysis_data[data_idx].get("analysis", {})
            
            # --- 1. 骨格 (Pose) の描画 ---
            pose_result = frame_analysis.get("pose", {})
            landmarks = pose_result.get("landmarks", [])
            if landmarks:
                # 線の描画
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        p1, p2 = landmarks[start_idx], landmarks[end_idx]
                        if p1["visibility"] > 0.5 and p2["visibility"] > 0.5:
                            c1 = (int(p1["x"] * width), int(p1["y"] * height))
                            c2 = (int(p2["x"] * width), int(p2["y"] * height))
                            cv2.line(frame, c1, c2, (255, 255, 255), 2)
                # 点の描画
                for lm in landmarks:
                    if lm["visibility"] > 0.5:
                        cx, cy = int(lm["x"] * width), int(lm["y"] * height)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # --- 2. 最大値 (Metrics) の表示 ---
            metrics = pose_result.get("metrics", {})
            max_speed = metrics.get("max_grip_speed", 0.0)
            max_sep = metrics.get("max_separation", 0.0)

            # テキスト背景（半透明の黒い矩形）
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # 最大値を表示
            cv2.putText(frame, f"MAX GRIP SPEED: {max_speed:>5.1f} km/h", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"MAX SEPARATION: {max_sep:>5.1f} deg", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2)

            # --- 3. バット (Bat) の描画 ---
            bat_result = frame_analysis.get("bat", {})
            detections = bat_result.get("detections", [])
            for det in detections:
                bbox = det.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path
