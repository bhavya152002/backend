import os
import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Scopes required
SCOPES = ['https://www.googleapis.com/auth/drive']

# Determine credential path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(BASE_DIR, os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "sheets-466006-60150f89fbb7.json"))

# Root folder for all recordings (Shared Drive or specific folder)
ROOT_FOLDER_ID = "1Vil-gKXQmDcDyqC7c3k6TSqG9oLkGq-g"

class DriveService:
    _service = None

    @classmethod
    def get_service(cls):
        if cls._service is None:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"[Drive] Credential file not found at: {CREDENTIALS_FILE}")
                return None
            
            try:
                creds = service_account.Credentials.from_service_account_file(
                    CREDENTIALS_FILE, scopes=SCOPES)
                cls._service = build('drive', 'v3', credentials=creds)
                print("[Drive] Service authenticated successfully.")
            except Exception as e:
                print(f"[Drive] Authentication failed: {e}")
                return None
        return cls._service

    @staticmethod
    def create_folder(folder_name, parent_id=None):
        service = DriveService.get_service()
        if not service: return None

        # Default to ROOT_FOLDER_ID if no parent specified
        if not parent_id:
            parent_id = ROOT_FOLDER_ID

        # Check if folder already exists
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        try:
            # Add shared drive support to search
            results = service.files().list(
                q=query, 
                fields="files(id, name)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                corpora="allDrives"
            ).execute()
            files = results.get('files', [])
            if files:
                return files[0]['id']
            
            # Create new folder
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]
                
            file = service.files().create(
                body=file_metadata, 
                fields='id',
                supportsAllDrives=True
            ).execute()
            print(f"[Drive] Created folder '{folder_name}' ID: {file.get('id')}")
            return file.get('id')
        except Exception as e:
            print(f"[Drive] Folder creation failed: {e}")
            return None

    @staticmethod
    def upload_file(file_path, folder_id=None, mime_type='video/mp4'):
        service = DriveService.get_service()
        if not service: return None
        
        file_name = os.path.basename(file_path)
        file_metadata = {'name': file_name}
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        
        try:
            print(f"[Drive] Uploading {file_name}...")
            # Add supportsAllDrives=True for Shared Drive compatibility
            file = service.files().create(
                body=file_metadata, 
                media_body=media, 
                fields='id, webViewLink',
                supportsAllDrives=True
            ).execute()
            print(f"[Drive] Upload complete. ID: {file.get('id')}")
            
            # Set permission to "anyone with link" (reader)
            # This is optional but good for sharing clips easily
            try:
                service.permissions().create(
                    fileId=file.get('id'),
                    body={'role': 'reader', 'type': 'anyone'},
                    fields='id',
                    supportsAllDrives=True
                ).execute()
            except Exception as e:
                print(f"[Drive] Permission setting failed (might be restricted by org policy): {e}")

            return file.get('webViewLink')
        except Exception as e:
            print(f"[Drive] Upload failed: {e}")
            return None

# Singleton instance access
drive_service = DriveService()
