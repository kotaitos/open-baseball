import argparse
import os
import glob
import json
import yaml
from baseball_lab.core.video import visualize_pose


def main():
    parser = argparse.ArgumentParser(
        description="等倍映像にスロー解析結果を同期させて可視化します。"
    )
    parser.add_argument(
        "--interim_dir", default="data/interim", help="中間データディレクトリ"
    )
    parser.add_argument(
        "--output_dir", default="data/processed", help="出力ディレクトリ"
    )
    parser.add_argument(
        "--player", default="Shohei_Ohtani", help="選手名 (ディレクトリ名として使用)"
    )

    args = parser.parse_args()
    
    # 設定の読み込み
    config_path = "analysis/analysis_config.yml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            if config_data and "analyzer" in config_data:
                config = config_data["analyzer"]

    player_interim_dir = os.path.join(args.interim_dir, args.player)
    player_output_base_dir = os.path.join(args.output_dir, args.player)

    video_dirs = [
        d for d in glob.glob(os.path.join(player_interim_dir, "*")) if os.path.isdir(d)
    ]

    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        json_path = os.path.join(video_dir, "analysis.json")
        trimmed_video_path = os.path.join(video_dir, "trimmed.mp4")
        meta_path = os.path.join(video_dir, "meta.json")

        if not all(
            os.path.exists(p) for p in [json_path, trimmed_video_path, meta_path]
        ):
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)
            slow_mo_factor = meta.get("slow_mo_factor", 1)

        # 出力ディレクトリ作成
        video_output_dir = os.path.join(player_output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        output_video_path = os.path.join(video_output_dir, "visualized.mp4")

        print(f"Visualizing {video_name} at original speed (Slow Factor: {slow_mo_factor}x)...")
        visualize_pose(
            trimmed_video_path,
            json_path,
            output_video_path,
            slow_mo_factor=slow_mo_factor,
            config=config,
        )
        print(f"Result saved to {output_video_path}")


if __name__ == "__main__":
    main()
