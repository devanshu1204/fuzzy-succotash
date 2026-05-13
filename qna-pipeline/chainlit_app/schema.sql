-- SQLite schema for chainlit.data.sql_alchemy.SQLAlchemyDataLayer.
-- Column names are camelCase and double-quoted to match the raw SQL the data
-- layer issues (see venv/.../chainlit/data/sql_alchemy.py — every query
-- quotes identifiers, e.g. `"threadId"`, `"createdAt"`).
--
-- All DDL is IF NOT EXISTS so this can be re-applied safely on every launch.

CREATE TABLE IF NOT EXISTS users (
    "id"          TEXT PRIMARY KEY,
    "identifier"  TEXT NOT NULL UNIQUE,
    "createdAt"   TEXT,
    "metadata"    TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id"              TEXT PRIMARY KEY,
    "createdAt"       TEXT,
    "name"            TEXT,
    "userId"          TEXT,
    "userIdentifier"  TEXT,
    "tags"            TEXT,
    "metadata"        TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    "id"             TEXT PRIMARY KEY,
    "name"           TEXT,
    "type"           TEXT,
    "threadId"       TEXT NOT NULL,
    "parentId"       TEXT,
    "streaming"      INTEGER,
    "waitForAnswer"  INTEGER,
    "isError"        INTEGER,
    "metadata"       TEXT,
    "tags"           TEXT,
    "input"          TEXT,
    "output"         TEXT,
    "createdAt"      TEXT,
    "start"          TEXT,
    "end"            TEXT,
    "generation"     TEXT,
    "showInput"      TEXT,
    "language"       TEXT,
    "indent"         INTEGER,
    -- Written by chainlit.step.Step.to_dict() with default False (not None), so
    -- they survive create_step()'s None-filter and must exist as columns or the
    -- INSERT raises and the step silently fails to persist.
    "defaultOpen"    INTEGER,
    "autoCollapse"   INTEGER,
    -- chainlit.message.Message.to_dict() emits these; usually None and filtered
    -- out, but kept here to match upstream writes if a /command is ever used.
    "command"        TEXT,
    "modes"          TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"        TEXT PRIMARY KEY,
    "forId"     TEXT NOT NULL,
    "threadId"  TEXT,
    "value"     INTEGER NOT NULL,
    "comment"   TEXT
);

CREATE TABLE IF NOT EXISTS elements (
    "id"            TEXT PRIMARY KEY,
    "threadId"      TEXT,
    "type"          TEXT,
    "url"           TEXT,
    "chainlitKey"   TEXT,
    "name"          TEXT,
    "display"       TEXT,
    "objectKey"     TEXT,
    "size"          TEXT,
    "page"          INTEGER,
    "language"      TEXT,
    "forId"         TEXT,
    "mime"          TEXT,
    "props"         TEXT
);

-- Indices for sidebar / thread loading performance.
CREATE INDEX IF NOT EXISTS idx_threads_userId      ON threads("userId");
CREATE INDEX IF NOT EXISTS idx_steps_threadId      ON steps("threadId");
CREATE INDEX IF NOT EXISTS idx_steps_parentId      ON steps("parentId");
CREATE INDEX IF NOT EXISTS idx_elements_threadId   ON elements("threadId");
CREATE INDEX IF NOT EXISTS idx_elements_forId      ON elements("forId");
CREATE INDEX IF NOT EXISTS idx_feedbacks_forId     ON feedbacks("forId");
CREATE INDEX IF NOT EXISTS idx_feedbacks_threadId  ON feedbacks("threadId");
