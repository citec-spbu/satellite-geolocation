from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Модель
    model_name: str = "convnext_tri"
    weights_dir: str = "weights"
    device: str = "cuda"
    image_size_h: int = 256
    image_size_w: int = 256

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "satellite_gallery"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "satellite-images"

    class Config:
        env_file = ".env"

_settings_instance = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance