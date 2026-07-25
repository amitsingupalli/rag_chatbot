from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000
    streamlit_port: int = 8501
    backend_url: str = "http://127.0.0.1:8000"

    data_dir: str = "./data"
    storage_dir: str = "./storage"

    chunk_size: int = 512
    chunk_overlap: int = 64
    similarity_top_k: int = 5
    rerank_top_n: int = 3
    memory_token_limit: int = 3900

    web_search_enabled: bool = True
    web_search_max_results: int = 5
    web_fetch_pages: bool = True
    web_fetch_top_n: int = 2
    web_page_max_chars: int = 4000

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def documents_path(self) -> Path:
        return self.data_path / "documents"

    @property
    def uploads_path(self) -> Path:
        return self.data_path / "uploads"

    @property
    def vector_store_path(self) -> Path:
        return self.data_path / "vector_store"

    @property
    def db_path(self) -> Path:
        return self.storage_path / "chat_history.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_path / "chroma_db"


settings = Settings()
