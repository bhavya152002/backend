import os
import datetime
from google.cloud import storage
from google.oauth2 import service_account

# Determine bucket name from env or use default
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "visionlogix-recordings")

# Determine credential path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Reusing the same service account JSON if it has GCS permissions,
# or user can specify a different one via GOOGLE_APPLICATION_CREDENTIALS
CREDENTIALS_FILE = os.path.join(BASE_DIR, os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "sheets-466006-60150f89fbb7.json"))

class GCSService:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"[GCS] Credential file not found at: {CREDENTIALS_FILE}")
                # Fallback to default auth if file missing (e.g., when running on GCP with service account)
                try:
                    cls._client = storage.Client()
                    print("[GCS] Client initialized using default credentials.")
                except Exception as e:
                    print(f"[GCS] Default authentication failed: {e}")
                    return None
            else:
                try:
                    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
                    cls._client = storage.Client(credentials=creds, project=os.getenv("GCP_PROJECT_ID"))
                    print("[GCS] Client authenticated successfully using service account file.")
                except Exception as e:
                    print(f"[GCS] Authentication via file failed: {e}")
                    return None
        return cls._client

    @staticmethod
    def upload_file(file_path, folder_name=None, mime_type='video/mp4'):
        """
        Uploads a file to GCS.
        folder_name: Optional 'directory' within the bucket.
        """
        client = GCSService.get_client()
        if not client: return None

        bucket = client.bucket(BUCKET_NAME)
        file_name = os.path.basename(file_path)
        
        # Construct blob path
        blob_path = f"{folder_name}/{file_name}" if folder_name else file_name
        blob = bucket.blob(blob_path)

        try:
            print(f"[GCS] Uploading {file_name} to {BUCKET_NAME}...")
            blob.upload_from_filename(file_path, content_type=mime_type)
            
            # Make the blob public? Or use signed URL?
            # For now, we'll try to make it public if bucket allows, otherwise signed URL is better for security.
            # User mentioned they are using it on localhost and cloudflare tunnel, typically public read is easiest for dev.
            try:
                blob.make_public()
                url = blob.public_url
                print(f"[GCS] Upload complete. Public URL: {url}")
            except Exception as e:
                # Fallback to signed URL if public access is restricted
                print(f"[GCS] Could not make public, generating signed URL: {e}")
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(hours=24),
                    method="GET",
                )
                print(f"[GCS] Signed URL generated.")

            return url
        except Exception as e:
            print(f"[GCS] Upload failed: {e}")
            return None

# Singleton instance
gcs_service = GCSService()
