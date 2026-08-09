import asyncio
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from rag_platform.core.config import get_settings
from rag_platform.db.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)
if config.config_file_name:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def configure(connection: object) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run() -> None:
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section) or {}, prefix="sqlalchemy."
    )
    async with engine.connect() as connection:
        await connection.run_sync(configure)
    await engine.dispose()


asyncio.run(run())
