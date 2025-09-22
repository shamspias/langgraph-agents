"""
Configuration settings for the LangGraph Agent System
"""
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation"""

    # API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    LANGSMITH_API_KEY: Optional[str] = Field(None, env="LANGSMITH_API_KEY")

    # Model Configuration
    DEFAULT_MODEL: str = Field("gpt-4.1-mini", env="DEFAULT_MODEL")
    EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    TEMPERATURE: float = Field(0.7, env="TEMPERATURE")

    # Vector Store Configuration
    VECTOR_DB_PATH: Path = Field(Path("./data/vector_store"), env="VECTOR_DB_PATH")
    COLLECTION_NAME: str = Field("knowledge_base", env="COLLECTION_NAME")
    CHUNK_SIZE: int = Field(500, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(50, env="CHUNK_OVERLAP")

    # Memory Configuration
    MEMORY_TYPE: str = Field("in_memory", env="MEMORY_TYPE")  # in_memory, sqlite, postgres
    MEMORY_DB_PATH: Path = Field(Path("./data/memory.db"), env="MEMORY_DB_PATH")

    # Search Configuration
    SEARCH_MAX_RESULTS: int = Field(5, env="SEARCH_MAX_RESULTS")

    # Tracing
    LANGSMITH_TRACING: bool = Field(False, env="LANGSMITH_TRACING")
    LANGSMITH_PROJECT: str = Field("langgraph-agents", env="LANGSMITH_PROJECT")

    # Agent Configuration
    MAX_ITERATIONS: int = Field(10, env="MAX_ITERATIONS")
    RECURSION_LIMIT: int = Field(25, env="RECURSION_LIMIT")

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
