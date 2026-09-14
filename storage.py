from __future__ import annotations
import boto3
from botocore.config import Config
from config import app_config, get_logger

logger = get_logger(__name__)

def _get_s3_client():
    """Khởi tạo S3 client tương thích với Cloudflare R2 hoặc AWS S3."""
    return boto3.client(
        "s3",
        endpoint_url=app_config.s3_endpoint_url,
        aws_access_key_id=app_config.s3_access_key,
        aws_secret_access_key=app_config.s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto" # Cloudflare R2 dùng region 'auto'
    )

def upload_file_to_cloud(workspace_id: str, file_name: str, file_bytes: bytes) -> None:
    """Đẩy file gốc lên Cloud Object Storage."""
    try:
        s3 = _get_s3_client()
        s3_key = f"workspaces/{workspace_id}/{file_name}"
        s3.put_object(
            Bucket=app_config.s3_bucket_name,
            Key=s3_key,
            Body=file_bytes,
        )
        logger.info("Đã upload file '%s' lên Cloud cho workspace '%s'.", file_name, workspace_id)
    except Exception as e:
        logger.exception("Lỗi khi upload file lên Cloud Storage: %s", e)

class MockUploadedFile:
    """Giả lập đối tượng Streamlit UploadedFile để truyền vào hàm render_preview."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data

    def seek(self, pos: int) -> None:
        pass

def get_cloud_files_for_preview(workspace_id: str) -> list[MockUploadedFile]:
    """Tải lại danh sách file từ Cloud Object Storage để hiển thị bản xem trước."""
    try:
        s3 = _get_s3_client()
        prefix = f"workspaces/{workspace_id}/"
        response = s3.list_objects_v2(Bucket=app_config.s3_bucket_name, Prefix=prefix)
        
        files = []
        for obj in response.get("Contents", []):
            s3_key = obj["Key"]
            file_name = s3_key.split("/")[-1]
            
            # Tải nội dung file bytes từ cloud về
            file_obj = s3.get_object(Bucket=app_config.s3_bucket_name, Key=s3_key)
            file_bytes = file_obj["Body"].read()
            files.append(MockUploadedFile(name=file_name, data=file_bytes))
            
        return files
    except Exception as e:
        logger.warning("Không thể lấy danh sách file từ Cloud Storage cho workspace '%s': %s", workspace_id, e)
        return []