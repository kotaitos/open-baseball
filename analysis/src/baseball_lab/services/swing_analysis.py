import os
import glob
import cv2
import json
import yaml
from baseball_lab.analyzers.swing import SwingAnalyzer

class SwingAnalysisService:
    def __init__(self, player_height_m: float = 1.93):
        # 1. 解析アルゴリズム設定 (analysis/analysis_config.yml)
        config_path = os.path.join(os.path.dirname(__file__), "../../../analysis_config.yml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data and "analyzer" in config_data:
                    config = config_data["analyzer"]

        # 2. 動画レジストリ（切り出し区間など - プロジェクトルート/video_registry.yml）
        registry_path = os.path.join(os.path.dirname(__file__), "../../../../video_registry.yml")
        self.video_registry = {}
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                self.video_registry = yaml.safe_load(f) or {}

        self.analyzer = SwingAnalyzer(player_height_m=player_height_m, config=config)
        self.player_height_m = player_height_m

    def _create_slow_mo_visual(self, video_path: str, slow_mo_factor: int, video_interim_dir: str, start_sec: float = 0.0, end_sec: float = None) -> str:
        """
        可視化用のスローモーション動画を指定区間（秒）で作成します。
        """
        os.makedirs(video_interim_dir, exist_ok=True)
        slow_path = os.path.join(video_interim_dir, "preprocessed.mp4")

        # 再生成を強制するために古いファイルは削除（設定変更の反映）
        if os.path.exists(slow_path):
            os.remove(slow_path)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(slow_path, fourcc, fps, (width, height))

        start_frame = int(fps * start_sec)
        end_frame = int(fps * end_sec) if end_sec is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or current_frame >= end_frame:
                break
            for _ in range(slow_mo_factor):
                out.write(frame)
            current_frame += 1
        cap.release()
        out.release()
        return slow_path

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

            # 1. スロー動画作成（可視化用）
            slow_path = self._create_slow_mo_visual(video_path, slow_mo, video_interim_dir, start_sec, end_sec)

            # 2. 解析（生データで行う - GEMINI.md 原則）
            output_path = os.path.join(video_interim_dir, "analysis.json")
            print(f"Analyzing RAW video ({start_sec}s - {end_sec}s): {video_path}...")
            # 生データなので speed_multiplier=1.0。
            # start_sec, end_sec を渡して必要な区間のみ解析する。
            self.analyzer.analyze_video(
                video_path,
                output_path,
                speed_multiplier=1.0,
                start_sec=start_sec,
                end_sec=end_sec,
            )

            with open(os.path.join(video_interim_dir, "meta.json"), "w") as f:
                json.dump({"slow_mo_factor": slow_mo}, f)
