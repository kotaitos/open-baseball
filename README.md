# Baseball-Lab

野球のスイング解析やパフォーマンス測定を行うための、動画解析・データ管理プラットフォームです。

## 🎯 概要
YouTubeから野球動画をダウンロードし、AIを用いて骨格解析を行い、解析結果を可視化した動画を生成・アップロードする一連のパイプラインを提供します。

## 📁 プロジェクト構成

```text
baseball-lab/
├── analysis/           # スイング解析のコアロジック（野球ドメインのライブラリ）
├── data/               # 動画データ、解析結果の保存先
│   ├── raw/            # ダウンロードしたオリジナル動画
│   ├── interim/        # 骨格座標データ（JSON）
│   └── processed/      # 骨格描画済みの加工動画
├── pipelines/          # パイプライン実行用スクリプト
└── README.md           # 本ドキュメント
```

## 🚀 使い方

詳細なセットアップ手順と実行方法は、[analysis/README.md](./analysis/README.md) を参照してください。

### クイックスタート

```bash
cd analysis
make setup  # 環境構築
make run    # 解析と可視化の実行
```

## 🛠️ 技術スタック
- **Environment**: [uv](https://docs.astral.sh/uv/)
- **Analysis**: Mediapipe, OpenCV, YOLO
- **External**: yt-dlp, Google API Client
