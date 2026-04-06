import os
import yaml
import urllib.request
import argparse
from pathlib import Path


def download_file(url, target_path):
    """
    指定されたURLからファイルをダウンロードする。
    """
    target_path = Path(target_path)
    if target_path.exists():
        # サイズの不一致などをチェックしても良い（今回は省略）
        return

    print(f"Downloading {url} to {target_path}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, open(target_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)
    print(f"Done: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="AIモデルの自動セットアップ")
    parser.add_argument("--config", default="models.yml", help="モデル構成ファイル")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    if not models:
        print("No models found in config.")
        return

    print(f"Checking {len(models)} models...")
    for model in models:
        name = model.get("name")
        url = model.get("url")
        path = model.get("path")

        if not url or not path:
            print(f"Skipping {name}: URL or path missing.")
            continue

        download_file(url, path)


if __name__ == "__main__":
    main()
