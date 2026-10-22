from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# Schema que o antigo `create_all()` (até a versão 7.0) deixava no banco.
LEGACY_CREATE_ALL_REVISION = "0003_aurora_control_plane"


class Base(DeclarativeBase):
    """Classe base dos modelos ORM."""


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, Any, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Leva o schema até a última migração. Alembic é o único caminho de schema."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    tables = set(inspect(engine).get_table_names())
    if tables and "alembic_version" not in tables:
        if "pipeline_runs" not in tables:
            raise RuntimeError(
                "Database has tables but no Alembic history and does not match the 7.0 schema. "
                "Back it up and remove it, or mark its revision with `alembic stamp`."
            )
        command.stamp(config, LEGACY_CREATE_ALL_REVISION)
    command.upgrade(config, "head")


def database_ok() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
