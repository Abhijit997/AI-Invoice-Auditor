"""Configuration settings for the API"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    api_title: str = "AI Invoice Auditor API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    incoming_folder: Path = base_dir / "data" / "incoming"
    processed_folder: Path = base_dir / "data" / "processed"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()