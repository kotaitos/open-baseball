import argparse
import yaml
import os
from baseball_lab.services.swing_analysis import SwingAnalysisService

def load_player_config(player_name: str):
    config_path = "players.yml"
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    player_data = config.get("players", {}).get(player_name)
    if player_data:
        return player_data
    
    return config.get("defaults", {})

def main():
    parser = argparse.ArgumentParser(
        description="生データを直接解析し、物理的に正しい速度を算出します。"
    )
    parser.add_argument("--input_dir", default="data/raw", help="入力動画ディレクトリ")
    parser.add_argument(
        "--output_dir", default="data/interim", help="出力中間データディレクトリ"
    )
    parser.add_argument("--slow_mo", type=int, default=4, help="可視化用のスロー倍率")
    parser.add_argument("--player", default="Shohei_Ohtani", help="選手名 (players.ymlのキー)")
    parser.add_argument(
        "--player_height", type=float, help="選手の身長 (メートル、指定があれば優先)"
    )

    args = parser.parse_args()

    # パスの構築
    player_input_dir = os.path.join(args.input_dir, args.player)
    player_output_dir = os.path.join(args.output_dir, args.player)

    if not os.path.exists(player_input_dir):
        print(f"Error: Player input directory not found at {player_input_dir}")
        return

    # YAMLから選手設定を読み込み
    player_config = load_player_config(args.player)
    
    height = args.player_height or player_config.get("height_m", 1.80)
    
    print(f"Analyzing for player: {args.player} (Height: {height}m)")

    service = SwingAnalysisService(player_height_m=height)
    service.run(input_dir=player_input_dir, output_dir=player_output_dir, slow_mo=args.slow_mo)


if __name__ == "__main__":
    main()
