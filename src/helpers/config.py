from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):

    GOOGLE_API_KEY:str
    FILE_ALLOWED_TYPES:str
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY :str
    LANGCHAIN_PROJECT :str
    LANGCHAIN_ENDPOINT : str = "https://api.smith.langchain.com"
    
    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
         extra="ignore"
    )


def get_settings():
    return Settings()