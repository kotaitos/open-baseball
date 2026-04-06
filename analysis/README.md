# Analysis Module

野球のスイング解析を行うためのコアロジックを格納するモジュールです。

## 📦 構成

- `baseball_lab.core.pose`: Mediapipeを使用した骨格推定ロジック
- `baseball_lab.core.video`: OpenCVを使用した動画描画・可視化ロジック
- `baseball_lab.services.swing_analysis`: スイング解析サービスの統合

## 🚀 セットアップ

`uv` を使用して依存関係を同期し、AIモデルを自動セットアップします。

```bash
# analysis ディレクトリで実行
make setup
```

これにより、以下の処理が自動的に行われます：
1. `uv sync` による Python 依存関係のインストール
2. `models.yml` に定義された AI モデル（Mediapipe 等）のダウンロードと配置

## 🏗️ モデル管理

使用する AI モデルは `models.yml` で管理されています。新しいモデルを追加する場合は、このファイルに URL と保存先パスを追記し、再度 `make setup` を実行してください。

## 🏃 実行方法

解析から可視化までを一括で実行するには `make run` を使用します。

```bash
make run SLOW_MO=4 PLAYER=Shohei_Ohtani
```

### 個別ステップの実行
詳細なオプションを指定する場合は、`pipelines/` のスクリプトを直接実行することも可能です。

```bash
# 例: 骨格解析のみ実行
uv run --project . ../pipelines/analyze_video.py --slow_mo 4 --player Shohei_Ohtani
```

## 🛠️ 技術スタック
- **Environment**: [uv](https://docs.astral.sh/uv/)
- **Analysis**: Mediapipe (Pose Landmarker), OpenCV
- **Logic**: One Euro Filter (ノイズ除去), 動的スケーリング (実距離算出)
