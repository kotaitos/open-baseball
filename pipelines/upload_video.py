import argparse
import os
import glob
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# スコープの定義
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_authenticated_service(client_secrets_file="client_secrets.json"):
    creds = None
    if os.path.exists("token.json"):
        from google.oauth2.credentials import Credentials

        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_video_to_youtube(
    youtube, file_path, title, description, category_id="17", privacy_status="private"
):
    """
    動画をYouTubeにアップロードします。
    """
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
        },
        "status": {"privacyStatus": privacy_status},
    }

    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=MediaFileUpload(file_path, chunksize=-1, resumable=True),
    )

    response = None
    while response is None:
        status, response = insert_request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    print(f"Video uploaded successfully! Video ID: {response['id']}")
    return response["id"]


def main():
    parser = argparse.ArgumentParser(
        description="加工済み動画をYouTubeにアップロードします。"
    )
    parser.add_argument(
        "--input_dir",
        default="data/processed",
        help="アップロードする動画のディレクトリ",
    )
    parser.add_argument(
        "--secrets",
        default="client_secrets.json",
        help="Google API クライアントシークレットファイル",
    )
    parser.add_argument(
        "--privacy",
        default="private",
        choices=["public", "private", "unlisted"],
        help="公開設定",
    )

    args = parser.parse_args()

    if not os.path.exists(args.secrets):
        print(
            f"Error: {args.secrets} not found. Please provide your Google API client secrets."
        )
        return

    video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return

    try:
        youtube = get_authenticated_service(args.secrets)

        for video_path in video_files:
            title = os.path.basename(video_path)
            description = "Analyzed by Baseball-Lab"

            print(f"Uploading {video_path}...")
            upload_video_to_youtube(
                youtube, video_path, title, description, privacy_status=args.privacy
            )

    except Exception as e:
        print(f"An error occurred during upload: {e}")


if __name__ == "__main__":
    main()
