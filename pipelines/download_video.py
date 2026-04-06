import argparse
import os
import yt_dlp


def sanitize_filename(filepath: str) -> str:
    """
    ファイル名をサニタイズ（スペースをアンダースコアに、ハッシュなどを除去）
    """
    base_dir = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)
    sanitized_name = "".join(
        [c if c.isalnum() or c in "._-" else "_" for c in file_name]
    )
    # 連続するアンダースコアを1つに
    while "__" in sanitized_name:
        sanitized_name = sanitized_name.replace("__", "_")

    sanitized_path = os.path.join(base_dir, sanitized_name)
    if filepath != sanitized_path:
        os.rename(filepath, sanitized_path)
    return sanitized_path


def download_youtube_video(url: str, output_dir: str) -> str:
    """
    YouTubeから動画をダウンロードし、保存されたファイルのパスを返します。
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        sanitized_path = sanitize_filename(raw_filename)

    return sanitized_path


def main():
    parser = argparse.ArgumentParser(description="YouTube動画をダウンロードします。")
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/shorts/_m1sxxpyGnc",
        help="YouTubeのURL",
    )
    parser.add_argument("--output_dir", default="data/raw", help="出力ディレクトリ")
    parser.add_argument(
        "--player",
        default="Shohei_Ohtani",
        help="選手名 (ディレクトリ名として使用)",
    )

    args = parser.parse_args()
    
    player_output_dir = os.path.join(args.output_dir, args.player)

    print(f"Downloading {args.url} to {player_output_dir}...")
    try:
        video_path = download_youtube_video(args.url, player_output_dir)
        print(f"Successfully downloaded: {video_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")


if __name__ == "__main__":
    main()

