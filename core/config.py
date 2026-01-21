"""
Application Configuration

Centralized configuration management for Bio-AI-SaaS.
Supports environment-based configuration with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "Bio-AI-SaaS Primer Design API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Database
    DATABASE_URL: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379"

    # LLM Configuration
    LLM_PROVIDER: str = "anthropic"  # anthropic or openai
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1

    # BLAST Configuration
    BLAST_DOCKER_IMAGE: str = "ncbi/blast:latest"
    BLAST_CONTAINER_NAME: str = "blast-worker"
    BLAST_DB_PATH: str = "/blast/blastdb"
    BLAST_TIMEOUT_SECONDS: int = 300

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 10
    RATE_LIMIT_PER_HOUR: int = 100
    RATE_LIMIT_SEQUENCES_PER_DAY: int = 1000
    MAX_SEQUENCE_LENGTH: int = 100000
    MAX_BATCH_SIZE: int = 100

    # Security
    ENABLE_BIOSECURITY_SCREENING: bool = True
    ENABLE_RATE_LIMITING: bool = True
    ENABLE_AUDIT_LOGGING: bool = True

    # Prompt Management
    PROMPT_VERSION: str = "v1.2.0-stable"
    PROMPTS_DIR: str = "prompts"

    # Report Generation
    TEMPLATES_DIR: str = "templates"
    REPORTS_DIR: str = "reports"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class Primer3Settings(BaseSettings):
    """Primer3-specific configuration."""

    # Salt concentrations (mM)
    MV_CONC: float = 50.0  # Monovalent cation (Na+)
    DV_CONC: float = 1.5   # Divalent cation (Mg2+)
    DNTP_CONC: float = 0.2  # dNTP
    DNA_CONC: float = 50.0  # Primer DNA (nM)

    # Thermodynamic model
    TM_METHOD: str = "santalucia"
    SALT_CORRECTION_METHOD: str = "santalucia"

    # Dimer threshold
    DIMER_DELTA_G_THRESHOLD: float = -9.0  # kcal/mol

    class Config:
        env_prefix = "PRIMER3_"


class VariantDBSettings(BaseSettings):
    """Variant database configuration."""

    DBSNP_VERSION: str = "b156"
    GNOMAD_VERSION: str = "v4.0"
    MAF_THRESHOLD: float = 0.01  # 1%

    # API endpoints (for remote databases)
    DBSNP_API_URL: Optional[str] = None
    GNOMAD_API_URL: Optional[str] = None

    class Config:
        env_prefix = "VARIANT_"


# Create settings instances
settings = Settings()
primer3_settings = Primer3Settings()
variant_db_settings = VariantDBSettings()


# ============================================================================
# Configuration Helpers
# ============================================================================

def get_database_url() -> str:
    """Get database URL with fallback to SQLite."""
    if settings.DATABASE_URL:
        return settings.DATABASE_URL
    return "sqlite:///./primer_design.db"


def get_redis_url() -> str:
    """Get Redis URL."""
    return settings.REDIS_URL


def get_llm_api_key() -> str:
    """Get API key for configured LLM provider."""
    if settings.LLM_PROVIDER == "anthropic":
        key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    else:
        key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

    if not key:
        raise ValueError(f"API key not configured for {settings.LLM_PROVIDER}")

    return key


def is_production() -> bool:
    """Check if running in production environment."""
    return settings.ENVIRONMENT == "production"


def get_log_level() -> str:
    """Get logging level."""
    if settings.DEBUG:
        return "DEBUG"
    return settings.LOG_LEVEL


# ============================================================================
# Directory Setup
# ============================================================================

def ensure_directories():
    """Ensure required directories exist."""
    dirs = [
        Path(settings.PROMPTS_DIR),
        Path(settings.TEMPLATES_DIR),
        Path(settings.REPORTS_DIR),
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_configuration():
    """Validate configuration on startup."""
    errors = []

    # Check LLM API key
    if settings.ENVIRONMENT != "development":
        try:
            get_llm_api_key()
        except ValueError as e:
            errors.append(str(e))

    # Check required directories
    if not Path(settings.PROMPTS_DIR).exists():
        errors.append(f"Prompts directory not found: {settings.PROMPTS_DIR}")

    if not Path(settings.TEMPLATES_DIR).exists():
        errors.append(f"Templates directory not found: {settings.TEMPLATES_DIR}")

    if errors:
        raise ConfigurationError(errors)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration errors: {'; '.join(errors)}")
