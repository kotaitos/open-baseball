import json
import os
from typing import List

import cv2

from baseball_lab.analyzers.base import BaseAnalyzer
from baseball_lab.core.pose import PoseAnalyzer
from baseball_lab.analyzers.bat import BatAnalyzer


class SwingAnalyzer:
    """
    ポーズ解析とバット解析を用いて、スイングを統合的に解析するクラス。
    """

    def __init__(
        self,
        analyzers: List[BaseAnalyzer] = None,
        player_height_m: float = 1.93,
        config: dict = None,
    ):
        if analyzers is None:
            self.analyzers = [
                PoseAnalyzer(player_height_m=player_height_m, config=config),
                BatAnalyzer(config=config),
            ]
        else:
            self.analyzers = analyzers

    def analyze_video(
        self,
        video_path: str,
        output_json_path: str,
        speed_multiplier: float = 1.0,
        start_sec: float = 0.0,
        end_sec: float = None,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # 動画自体のFPSに倍率をかけて、実時間での「有効なFPS」を算出
        fps = cap.get(cv2.CAP_PROP_FPS)
        effective_fps = fps * speed_multiplier

        # 指定開始位置までシーク
        start_frame = int(fps * start_sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 終了位置の決定
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = (
            min(int(fps * end_sec), total_frames) if end_sec is not None else total_frames
        )

        results_data = []
        frame_idx = 0  # 解析対象のセグメント内でのインデックス（0から開始）

        while cap.isOpened():
            ret, frame = cap.read()
            # 現在の読み込みフレーム番号を取得
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not ret or curr_pos > end_frame:
                break

            # タイムスタンプはセグメントの開始(start_sec)を0として計算
            frame_result = {
                "frame": frame_idx,
                "timestamp_ms": int(1000 * frame_idx / effective_fps),
                "analysis": {},
            }
            context = {}

            # 各アナライザーの実行（実時間ベースのFPSを渡す）
            for analyzer in self.analyzers:
                result = analyzer.analyze_frame(
                    frame, frame_idx, effective_fps, context=context
                )
                frame_result["analysis"][analyzer.name] = result
                context[analyzer.name] = result

            results_data.append(frame_result)
            frame_idx += 1

        cap.release()

        # 終了処理（クリーンアップが必要なアナライザー用）
        for analyzer in self.analyzers:
            if hasattr(analyzer, "close"):
                analyzer.close()

        # 結果の保存
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(results_data, f, indent=2)

        return output_json_path
