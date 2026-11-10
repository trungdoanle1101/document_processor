# Working with Claude on this repository

## The project

A knowledge-base backend for restricted Vietnamese regulations: upload documents, find their
structure (Chương, Điều, khoản…), chunk them, index them in a vector store, and search them
with access control by group.

- `src/document_processor/`: the framework (file adapters, cleaning, profiles, outlines,
  chunking, review pages, retrieval). Tests in `tests/`.
- `apps/server/` (`kb_server`): FastAPI + SQLAlchemy 2.0 + Postgres + Alembic.
  `api → services → domain → storage`, plus `indexing/` for the worker. Services check
  permissions and raise; repositories only report. Job handlers never touch the database.
- `apps/cli/`: an HTTP-only client, later.

The design, with the reason for each decision and the build order:
@docs/backend-design.md

## Commands

```bash
uv sync --all-packages                        # install everything
docker compose -f docker/docker-compose.yml up -d postgres
docker compose -f docker/docker-compose.yml exec postgres createdb -U kb kb_test   # once

# server tests: run from apps/server (its tests/ package clashes with the framework's)
cd apps/server && KB_TEST_DATABASE_URL=postgresql+psycopg://kb:kb@localhost/kb_test uv run pytest
#   without KB_TEST_DATABASE_URL a throwaway Postgres container is started (slower)

uv run pytest tests                           # framework tests, from the root
uv run ruff check . && uv run ruff format --check .
uv run mypy src apps/server/src apps/server/tests

cd apps/server && uv run alembic upgrade head # migrations (settings from apps/server/.env)
uv run kb-server create-admin --email … --name …
uv run kb-server serve --reload
```

## Roles

- I write the code. Claude designs, explains, reviews and writes tests when asked.
- Show only the files that change. Answer briefly when I ask for brevity.
- Never commit or push to this repository. Review from a separate checkout.

## How to present a design

1. **Problem first:** describe the situation and what goes wrong without a design,
   before showing any structure.
2. **Caller first:** show how a piece will be used before describing the piece.
3. **Let me sketch:** give me the problem, let me write a rough version, then refactor
   it together.
4. **Explain why** each part exists: the problem it solves, not only what it is.
5. **Label parameters by role** when introducing a function:
   - the thing being worked on (*what?*), positional, first;
   - the settings (*how?*), keyword-only;
   - the tools (*with what?*), keyword-only.

   Keep to about three or four parameters.
6. **Name patterns** (repository, idempotent consumer, outbox, …) when they come up,
   problem first.

## Naming

- **The call-site test:** reading the line where a name is used, without opening the
  definition, should say what happens. Name things by what a reader sees them do, not
  by where they sit in the design.
- One word per idea everywhere (e.g. `folder`, never also `prefix` or `dir`).
- Functions are verbs, values are nouns, booleans read as questions, units go in names.
- Give a one-line reason whenever a proposed name isn't self-explanatory.

## Reviews

- Order: **must fix**, **should fix**, **nits**.
- Run ruff, mypy and pytest, and try the behaviour on real input before reporting.

## Code style

- ruff format, ruff check and strict mypy settle the mechanical details.
- Each function stays at one level of detail: the top function reads as steps, details
  live in helpers below it.
- Return early instead of nesting.
- Use types to rule out wrong states: frozen dataclasses, `Literal`, `uuid.UUID` for ids.
  Check input where it enters, then trust it.
- Docstrings give the contract (first line: what it does or returns). Comments explain
  why, on their own line above the code. Rewrite a docstring when copying a function.
- Catch narrowly; raise where you know what's wrong, with a message a person can act on.
- Order files like a newspaper: public names first, helpers below in calling order.
- Test names are sentences describing behaviour.

## Security (the repository is public)

- Never read anything under `data/`.
- The regulation documents are restricted. Never commit documents or question sets;
  keep them under the git-ignored `data/`. Never send them to cloud services without
  explicit approval.
- Never commit `.env`. Never log tokens; store only their hashes.
- Log ids, versions and codes only, never document text.
- Keep code database-agnostic: no driver-specific internals.
