from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = None
    gemini_model: str = "models/gemini-flash-latest"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    llama_cloud_api_key: str | None = None

    embedding_provider: str = "gemini"
    embedding_model: str = "models/text-embedding-004"

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000
    streamlit_port: int = 8501
    backend_url: str = "http://127.0.0.1:8000"

    data_dir: str = "./data"
    storage_dir: str = "./storage"

    chunk_size: int = 1024
    chunk_overlap: int = 204
    similarity_top_k: int = 10
    rerank_top_n: int = 5
    max_tokens: int = 4096
    memory_token_limit: int = 3900

    web_search_enabled: bool = True
    web_search_max_results: int = 5
    web_fetch_pages: bool = True
    web_fetch_top_n: int = 2
    web_page_max_chars: int = 4000

    @property
    def get_gemini_api_keys(self) -> list[str]:
        import re
        keys = []
        if self.gemini_api_key:
            parts = re.split(r'[,\n]', str(self.gemini_api_key))
            for p in parts:
                clean = p.strip(' "\' \t')
                if clean and clean not in keys:
                    keys.append(clean)
        return keys

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
