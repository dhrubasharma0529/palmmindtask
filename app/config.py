from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    groq_api_key:str
    # openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    gemini_api_key: str
    gemini_embedding_model: str = "gemini-embedding-001"
    groq_chat_model : str = "llama-3.3-70b-versatile"
  
    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "documents"

    # Redis
    redis_host: str
    redis_port: int
    redis_password: str

    # MySQL (via MySQL Workbench)
    database_url: str


settings = Settings()
