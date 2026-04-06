import glob
import json
import os
import subprocess

import cv2
import yaml

from baseball_lab.analyzers.swing import SwingAnalyzer


class SwingAnalysisService:
    def __init__(self, player_height_m: float = 1.93):
        # 1. 解析アルゴリズム設定 (analysis/analysis_config.yml)
        config_path = os.path.join(
            os.path.dirname(__file__), "../../../analysis_config.yml"
        )
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data and "analyzer" in config_data:
                    config = config_data["analyzer"]

        # 2. 動画レジストリ（切り出し区間など - プロジェクトルート/video_registry.yml）
        registry_path = os.path.join(
            os.path.dirname(__file__), "../../../../video_registry.yml"
        )
        self.video_registry = {}
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                self.video_registry = yaml.safe_load(f) or {}

        self.analyzer = SwingAnalyzer(player_height_m=player_height_m, config=config)
        self.player_height_m = player_height_m

    def _prepare_videos(
        self,
        video_path: str,
        slow_mo_factor: int,
        video_interim_dir: str,
        start_sec: float = 0.0,
        end_sec: float = None,
    ) -> tuple[str, str]:
        """
        ffmpegを使用して、可視化用の等倍動画と、解析用の補間スローモーション動画を作成します。
        """
        os.makedirs(video_interim_dir, exist_ok=True)
        trimmed_path = os.path.join(video_interim_dir, "trimmed.mp4")
        slow_path = os.path.join(video_interim_dir, "preprocessed.mp4")

        # 動画情報を取得
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        duration_cmd = []
        if end_sec is not None:
            duration_cmd = ["-t", str(end_sec - start_sec)]

        # 1. 等倍切り出し動画の作成
        cmd_trimmed = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
        ] + duration_cmd + [
            "-i", video_path,
            "-c:v", "libx264", "-crf", "18",
            trimmed_path
        ]
        
        # 2. 補間スロー動画の作成 (minterpolateを使用)
        # mi_mode=blend は高速で、速度計算の物理的一貫性を保つのに適しています
        target_fps = fps * slow_mo_factor
        cmd_slow = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
        ] + duration_cmd + [
            "-i", video_path,
            "-vf", f"minterpolate=fps={target_fps}:mi_mode=blend",
            "-c:v", "libx264", "-crf", "18",
            slow_path
        ]

        print(f"Creating trimmed video: {' '.join(cmd_trimmed)}")
        subprocess.run(cmd_trimmed, check=True, capture_output=True)
        
        print(f"Creating interpolated slow-mo video ({target_fps} fps): {' '.join(cmd_slow)}")
        subprocess.run(cmd_slow, check=True, capture_output=True)

        return trimmed_path, slow_path

    def run(self, input_dir: str, output_dir: str, slow_mo: int):
        video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
        video_files = [f for f in video_files if "_slow_" not in f]

        # 動画リストの取得
        video_configs = self.video_registry.get("videos", {})
        registry_defaults = self.video_registry.get("defaults", {})

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_interim_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_interim_dir, exist_ok=True)

            # 切り出し秒数の決定（video_registry.yml から取得）
            settings = video_configs.get(video_name, registry_defaults)
            start_sec = settings.get("start_sec", 0.0)
            end_sec = settings.get("end_sec", None)

            # 1. 解析用スロー動画と可視化用等倍動画の作成
            trimmed_path, slow_path = self._prepare_videos(
                video_path, slow_mo, video_interim_dir, start_sec, end_sec
            )

            # 2. 解析（精度向上のため補間スロー動画で行う）
            output_path = os.path.join(video_interim_dir, "analysis.json")
            print(f"Analyzing interpolated video for accuracy: {slow_path}...")
            # 動画自体が高密度FPS(例:120fps)で保存されているため、multiplierは1.0で物理的に正しくなる
            self.analyzer.analyze_video(
                slow_path,
                output_path,
                speed_multiplier=1.0,
            )

            with open(os.path.join(video_interim_dir, "meta.json"), "w") as f:
                json.dump({"slow_mo_factor": slow_mo}, f)
