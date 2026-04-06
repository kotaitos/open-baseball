# Pipelines

各ステップを実行するためのエントリポイントとなるスクリプト群です。

## 📜 スクリプト一覧

- `download_video.py`: YouTube URLから `data/raw` へ動画をダウンロードします。
- `analyze_video.py`: `data/raw` の動画を解析し、`data/interim` に骨格座標（JSON）を出力します。
- `visualize_video.py`: `data/raw` と `data/interim` を統合し、`data/processed` に可視化動画を出力します。
- `upload_video.py`: `data/processed` の動画を自身のYouTubeチャンネルにアップロードします。

## 🏃 実行方法
プロジェクトルートから `uv run` を使用して実行することを推奨します。

```bash
uv run --project analysis python pipelines/<script_name>.py
```
