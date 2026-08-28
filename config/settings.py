from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    riot_api_key: str
    henrik_api_key: str = ""
    riot_region: str = "na"
    riot_cluster: str = "americas"
    data_dir: str = "data"

    @field_validator("riot_region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        valid = {"na", "eu", "ap", "kr", "br", "latam"}
        if v not in valid:
            raise ValueError(f"riot_region must be one of {valid}")
        return v

    @field_validator("riot_cluster")
    @classmethod
    def validate_cluster(cls, v: str) -> str:
        valid = {"americas", "europe", "asia", "esports"}
        if v not in valid:
            raise ValueError(f"riot_cluster must be one of {valid}")
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
