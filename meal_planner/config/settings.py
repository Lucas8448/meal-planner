"""Configuration settings for the application."""

import os
from typing import Optional, Set
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    kassalapp_api_key: str = os.getenv("kassalapp_api_key", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Kassalapp API
    kassalapp_base_url: str = "https://kassal.app/api/v1"
    
    # Location settings
    location_latitude: Optional[str] = os.getenv("location_latitude")
    location_longitude: Optional[str] = os.getenv("location_longitude")
    location_radius: Optional[str] = os.getenv("location_radius")
    
    # LLM settings
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 1.0
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # For logging
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": ".env"
    }


# Global settings instance
settings = Settings()

# Global caches
_nearby_store_groups_cache: Optional[Set[str]] = None


def validate_required_settings():
    """Validate that required settings are present."""
    if not settings.kassalapp_api_key:
        raise ValueError("Kassalapp API key missing in environment variables.")
    
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key missing in environment variables.")
    
    if not all([settings.location_latitude, settings.location_longitude, settings.location_radius]):
        print("Warning: Location environment variables not fully set. Nearby store filtering may fail.") 