"""Local SQLite persistence for Chainlit threads / steps / messages.

Wires up `chainlit.data.sql_alchemy.SQLAlchemyDataLayer` against a SQLite DB
under `qna-pipeline/chainlit_data/chat_history.db` so conversations survive
`chainlit run` restarts and re-appear in the sidebar on relaunch.

Chainlit ships no migrations — every query in `chainlit/data/sql_alchemy.py`
uses raw `text()` against a hand-authored schema. We apply our own schema
(adjacent `schema.sql`) on import; every statement is `IF NOT EXISTS`, so the
bootstrap is idempotent.

No element/file storage client is configured — our UI emits text Messages and
tool Steps only. Chainlit will log a single warning at start-up about that;
ignore it.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

log = logging.getLogger(__name__)


_THIS_DIR = Path(__file__).resolve().parent
_SCHEMA_PATH = _THIS_DIR / "schema.sql"
_DB_DIR = _THIS_DIR.parent / "chainlit_data"
DB_PATH = _DB_DIR / "chat_history.db"


_STEPS_REQUIRED_COLUMNS: tuple[tuple[str, str], ...] = (
    # `chainlit.step.Step.to_dict()` always emits these with default `False`
    # (not None) so they survive `create_step()`'s None-filter. If the column
    # is missing the INSERT raises, the step silently fails to persist (it's
    # fire-and-forget via `asyncio.create_task`), and the thread reload shows
    # only the top-level user/assistant messages — with orphan `parentId`s
    # pointing at the dropped tool/LLM steps, which breaks the UI tree.
    ("defaultOpen", "INTEGER"),
    ("autoCollapse", "INTEGER"),
    # Emitted by `chainlit.message.Message.to_dict()`; usually None and thus
    # filtered out, but added for parity if a /command is ever used.
    ("command", "TEXT"),
    ("modes", "TEXT"),
)


def _migrate_steps_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute('PRAGMA table_info("steps")')}
    for name, sqltype in _STEPS_REQUIRED_COLUMNS:
        if name in existing:
            continue
        conn.execute(f'ALTER TABLE steps ADD COLUMN "{name}" {sqltype}')
        log.info("chainlit data layer: added steps.%s column", name)


def _bootstrap_schema_sync(db_path: Path, schema_path: Path) -> None:
    """Create the Chainlit tables in `db_path` if they don't exist yet.

    Runs synchronously via stdlib `sqlite3` so the DB is fully provisioned
    before Chainlit's async data layer issues its first query at request time.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ddl = schema_path.read_text(encoding="utf-8")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(ddl)
        _migrate_steps_columns(conn)
        conn.commit()
    finally:
        conn.close()
    log.info("chainlit data layer: schema ensured at %s", db_path)


# Apply schema at import time so the DB exists before Chainlit's first query.
_bootstrap_schema_sync(DB_PATH, _SCHEMA_PATH)


@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    return SQLAlchemyDataLayer(
        conninfo=f"sqlite+aiosqlite:///{DB_PATH}",
        show_logger=False,
    )
