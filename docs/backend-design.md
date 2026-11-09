# Knowledge base backend: design guide

The shape of the backend, the CLI and how they sit on `document_processor`. Each decision
comes with the reason for it; section 15 lists the ones that are easy to undo by accident.

Status markers: **(built)** exists on the `apps` branch; **(next)** is being worked on;
everything else is planned.

---

## 1. Repository

```text
pyproject.toml              uv workspace root; document_processor (the framework) stays here
src/document_processor/     the framework: adapters, cleaning, profiles, chunking, preparation, retrieval
tests/                      the framework's tests
apps/
  server/                   package kb-server → document_processor (workspace), FastAPI, SQLAlchemy,
                            Alembic, psycopg, pydantic-settings, argon2-cffi, boto3
  cli/                      package kb-cli → httpx, Typer only (no server or framework imports)
docker/docker-compose.yml   postgres (built); qdrant, minio later
data/                       git-ignored: restricted documents and question sets, never committed
```

## 2. Server package

```text
apps/server/
  pyproject.toml            scripts: kb-server = "kb_server.manage:main"
  alembic.ini
  migrations/               env.py (URL from Settings, target_metadata = Base.metadata), versions/
  src/kb_server/
    settings.py             Settings (pydantic-settings, env prefix KB_, .env next to where it runs)
    wiring.py               Services: builds everything the API needs
    app.py                  create_app(services) → FastAPI
    manage.py               kb-server create-admin | serve (built); migrate | worker | reprepare | gc
    api/                    routers + request/response schemas; HTTP only
    services/               use cases and permission checks; no HTTP, no SQL
    domain/                 frozen dataclasses and errors: User, Principal, Document, StoredFile, …
    storage/                SQL and files only
      database.py  models.py  repositories/  files.py
    indexing/               the worker's side (models load only here)
      prepared.py           PreparationSettings, PreparedParts, save/load/delete prepared output (built)
      handlers.py           ProfileChooser, ClaimedJob, run_prepare_job (built), run_index_job (next)
  tests/                    api/ services/ storage/ indexing/ domain/
```

### Layering

- `api` → `services` → `domain` and `storage`; `indexing` → `storage` and the framework. Nothing
  imports upward. Why: each layer can be tested without the ones above it.
- Routers convert `XIn` → arguments, call one service method, convert the result → `XOut`.
- Services check permissions and raise domain errors. Repositories only report (`None`, `bool`,
  counts); they never raise for "not found". Why: one place decides what's an error and which one.
- Repositories take a `Session` and return domain dataclasses, never rows.
- One transaction per request: commit on return, roll back on raise.
- Job handlers never touch the database: they return a result, and the worker records it. Why:
  handlers can be tested with only a file store.

### Kinds of objects

| Kind | Use for | Naming |
|---|---|---|
| Pydantic `BaseModel` | HTTP bodies | `DocumentIn`, `DocumentOut` |
| Pydantic `BaseSettings` | configuration from the environment | `Settings` |
| frozen `@dataclass` | domain values, results | `Document`, `Principal`, `PrepareResult` |
| SQLAlchemy model | tables, inside `storage/` only | `DocumentRow` |
| class | services, repositories, stores | `DocumentService`, `JobRepository` |
| `Protocol` | swappable parts | `FileStore`, the framework's `Chunker`, `Embedder` |
| function | pure helpers | `original_key`, `prepared_version_folder` |

### Parameters by role

A function's parameters usually play three roles: the thing being worked on (first,
positional), the settings and the tools (keyword-only). `run_prepare_job(job, *, chooser,
store)`: this job, prepared this way, with this store.

---

## 3. Settings (environment)

```text
built      KB_DATABASE_URL  KB_SESSION_TTL_HOURS  KB_ACT_FOR_HEADERS  KB_MAX_UPLOAD_MB
           KB_STORAGE_KIND (local | s3)  KB_STORAGE_PATH
worker     KB_PROFILES (candidates, most specific first)  KB_FALLBACK_PROFILE (plain)
           KB_CHUNK_TOKENS  KB_EMBEDDER_SPEC  KB_EMBEDDER_FOLDER  KB_EMBEDDER_DEVICE
           KB_QDRANT_URL  KB_QDRANT_COLLECTION
           KB_WORKER_POLL_SECONDS  KB_JOB_MAX_ATTEMPTS  KB_JOB_TIMEOUT_MINUTES  KB_LOG_LEVEL
later      KB_S3_ENDPOINT  KB_S3_BUCKET  KB_S3_ACCESS_KEY  KB_S3_SECRET_KEY
           KB_OLLAMA_URL  KB_CHAT_MODEL  KB_CHAT_CONTEXT_SIZE  KB_CHAT_TIMEOUT
           KB_RERANKER_SPEC  KB_RERANKER_FOLDER  KB_RERANKER_DEVICE  KB_RERANK_DEPTH
           KB_ANSWER_PASSAGES  KB_MIN_SCORE  KB_MIN_PASSAGE_SCORE  KB_SECTION_CHARS  KB_HISTORY_TURNS
```

`.env` holds the database password: it is git-ignored and never committed.

---

## 4. Database

- SQLAlchemy 2.0 ORM, synchronous, psycopg 3. Code stays database-agnostic: no driver-specific
  internals.
- Alembic for every schema change; constraint naming convention on `MetaData`.
- Tests migrate a real Postgres once with Alembic, then run each test in a transaction that is
  rolled back.
- Ids: UUIDs. Timestamps: `timestamptz`.
- Columns are added by migration when a feature first uses them, not ahead of time.

### Tables

```text
built
users                 id, email (unique, lower-case), display_name, password_hash, role (reader|manager|admin),
                      is_active, created_at
groups                id, name (unique, lower-case), created_at      (`everyone` is seeded; new users join it)
user_groups           user_id, group_id                              pk (user_id, group_id)
credentials           id, user_id, kind (session|api_key), name, prefix, token_hash (unique),
                      act_for_users, created_at, last_used_at, expires_at, revoked_at
collections           id, slug (unique), title, description, created_at
collection_groups     collection_id, group_id (restrict)             default read groups for new documents
collection_managers   collection_id, user_id
stored_files          id, sha256 (unique), size, media_type, storage_key, created_at
documents             id, collection_id (restrict), title, status (draft|active), current_version,
                      created_by, created_at, updated_at
document_versions     document_id, version, file_id (restrict), original_filename, provenance (json, nullable),
                      uploaded_by, created_at                        pk (document_id, version)
document_groups       document_id, group_id (restrict)               who may read it
indexing_jobs         id, kind (prepare|index|sync_attributes), document_id (cascade), version (nullable),
                      status (queued|running|done|failed), attempts, error, created_by,
                      created_at, started_at, finished_at            index (status, created_at)

next
documents.status      + deleting
indexing_jobs.kind    + delete

later
document_preparation  document_id (pk), profile, set_by, set_at      a manager's profile override
tags, document_tags, document_attributes
document_relations    source_id, target_id, kind (replaces|amends|guides|appendix_of|based_on)
retirements, retirement_documents
upload_batches, batch_groups
answers, feedback
```

There is no `profile` column on `documents`: the profile is chosen automatically when
preparing (section 7), and an override lives in its own table. Why: most documents never need
one, and the override is a decision with its own author and time.

### Rules

- Read access: a reader sees a document when they share a group with it. Managers of its
  collection and admins see all of the collection's documents. A document with no groups is
  visible only to them; public means the `everyone` group.
- A document's own groups are what's checked; a collection's groups are only copied to new
  documents. New document's groups: `group_ids` given → those; left out → the collection's
  defaults; `private=true` → none.
- Readers see a collection when they share one of its default groups. A hidden collection or
  document is a 404, never a 403. Why: a 403 would confirm that it exists.
- Managing a collection needs the manager role and a `collection_managers` row (admins manage
  all; a front end acting for someone manages nothing). Uploading needs `can_manage`.
- A collection with documents can't be deleted; a group in use can't be deleted (`restrict`).
- The same file (sha256) as a document's current version in the same collection → 409
  `duplicate_file` with that document's id.
- A single upload is `active` at once; `draft` is for batches.
- Changing title, groups, tags or attributes → a `sync_attributes` job, no re-embedding.
- A new file → a new version → a `prepare` job. A profile override → a `prepare` job for the
  current version.
- Deleting a document removes it everywhere (section 8, "Deleting").

---

## 5. Index (Qdrant): chunk attributes

```text
built in run_index_job (next)   access_group_ids   collection_id   document_version
later                           tags   status   retired_on   part
```

- The index holds one version per document: indexing replaces all of the document's chunks.
- Search filters inside the index: `access_group_ids` contains one of the reader's groups.
  Why: filtering only after the search loses results and reveals that hidden hits exist.
  A private document has no groups, so no group filter matches it.
- After retrieval: drop hits whose `document_version` isn't the document's current version, or
  whose document is no longer `active`.

---

## 6. File storage

```python
class FileStore(Protocol):
    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None: ...
        # keeps whatever is already at the key
    def open(self, key: str) -> BinaryIO: ...          # missing → FileNotFoundError
    def exists(self, key: str) -> bool: ...            # files only
    def delete_folder(self, folder: str) -> int: ...   # every key under folder + "/"
```

- A folder is a key without its last `/name` part. On S3 a folder exists only as the start
  of its files' keys.
- `LocalFileStore` (built) writes atomically: a temporary file in the same folder, `fsync`,
  `os.replace`. It refuses keys outside its root. `S3FileStore` later, chosen by
  `KB_STORAGE_KIND`.

### Layout

```text
originals/sha256/<aa>/<bb>/<sha256>          the uploaded file, addressed by its content
prepared/<document_id>/v<version>/           one version's prepared output
    logical.json  chunks.jsonl  review.html
```

- Originals are content-addressed: the same bytes are stored once, and a key can't point at
  the wrong content. Several versions may share one original, so it's deleted only when no
  `stored_files` row needs it.
- Prepared output belongs to its version, not to the file's content. Why: it retires with the
  document (no cache that grows forever), and nothing is silently reused after settings or
  code change. Re-preparing replaces the folder: `save_prepared` deletes it first, so a save
  that stops halfway loads as nothing, never as old and new files mixed.
- Only `indexing/prepared.py` knows this layout: callers pass a document id and version.

### Upload order (built)

```text
1. check the caller may manage the collection (before reading the file)
2. stream to a temporary file: sha256 + size, enforce KB_MAX_UPLOAD_MB → 413
3. check the type by content: .docx (a zip with word/document.xml) or .txt (UTF-8, no NUL) → else 415
4. duplicate check → 409
5. FileStore.put
6. one transaction: stored_files (if new) → documents → document_versions → document_groups
   → indexing_jobs (prepare)
→ 201 DocumentOut with the queued job
```

The file is written before the transaction: a failed transaction leaves an unused file, never a
row without its file. Writing the document and its job in one transaction means there is never
a document without its job (the outbox idea).

---

## 7. Preparation and profiles

- Profiles describe how a kind of document is structured. Built in: `vn_regulation`
  (Chương, Điều, khoản…) and `plain` (no levels: chunked by size only).
- **Automatic choice:** `KB_PROFILES` lists candidates; each is tried in order, and the first
  whose outline has a section wins. If none does, `KB_FALLBACK_PROFILE` (`plain`) is used.
  List candidates from most specific to most general.
- First-match is right while there is one structured profile. When a second arrives, switch to
  scoring every candidate (coverage, numbering in sequence, sensible depth, warnings), checked
  against a small labelled set of documents. List order then only breaks ties.
- `ProfileChooser.from_profiles(...)` is the only way to build the chooser, so its preparers and
  its `PreparationSettings` always agree. Unknown profile names fail at worker startup.
- `PreparationSettings`: candidate profiles and fallback (`name@version`), the chunker (its type
  and options, without the tokenizer), and the embedder's identity (whose tokenizer sized the
  chunks).
- Each preparation saves a review page (`review.html`) next to its parts, for a person to check
  the outline and chunks.

---

## 8. The worker and its jobs

### Rules that keep it safe

The worker writes to three places that share no transaction: Postgres, the file store and the
index. These rules make a crash at any point harmless:

1. Every step is safe to repeat (idempotent consumer): delete-then-write, put-if-absent,
   replace-all-chunks. A job may run more than once (at-least-once delivery).
2. The database is the truth; prepared output and the index can be rebuilt from it.
3. Do the work first, then record it in one transaction.
4. Order writes so a crash leaves something harmless: an orphaned file, never a row pointing
   at nothing.
5. One worker: jobs never race. A change during a job queues a follow-up job, which runs after.

### Decisions

- W1: separate `prepare` (no models) and `index` (embeds) jobs; prepared output is saved
  between them (a claim check: the index job loads it by reference).
- W2: index attributes are `access_group_ids`, `collection_id`, `document_version` (section 5).
- W3: retries up to `KB_JOB_MAX_ATTEMPTS`; `PermanentJobError` fails a job at once.
- W4: one worker; a job `running` longer than `KB_JOB_TIMEOUT_MINUTES` is requeued at startup
  (a lease that expired).
- W5: models load only in the worker (`kb-server[worker]` extra, `IndexTools`).
- W6: a superseded job (its version isn't current, or the document is `deleting`) is marked done
  without doing anything.

### Prepare job (built)

```text
1. media type → extension (.docx / .txt); anything else → PermanentJobError
2. copy the original into a temporary folder as document.<ext>; missing → PermanentJobError
3. ProfileChooser.prepare: OSError → retry; any other error → PermanentJobError
   (preparation gives the same result for the same bytes and code)
4. save_prepared, then review.html
5. log warning counts (ids and codes only); return PrepareResult
```

`PrepareResult`: profile, profiles without structure, settings, and whether to index next (the
document is `active`). The worker then, in one transaction: sets the version's provenance to
`{"preparation": …}` (dropping any old `index` part), marks the job done, and queues `index`.

### Index job (next)

```text
1. the chunks were sized for this embedder (provenance settings.embedder == embedder.identity),
   else PermanentJobError; checked first, so a refused job loads nothing
2. load_prepared; missing → PermanentJobError ("prepare it again")
3. attributes: access_group_ids (sorted), collection_id, document_version
4. index_document: replaces all of the document's chunks; no chunks removes it
5. log the chunk count; return IndexResult
```

Groups are read when the job is claimed, so the newest groups are indexed. Errors from the
index or the model pass through and are retried. The worker then writes `provenance["index"]`
and marks the job done.

### Provenance

Where a version's prepared output and vectors came from: the inputs and the decisions, not the
results (counts and warnings are rebuilt from the files when shown).

```json
{
  "preparation": {"profile": "vn_regulation@4", "profiles_without_structure": [], "settings": {"...": "..."}},
  "index": {"embedder": "huggingface:BAAI/bge-m3@<revision>", "index": "qdrant:kb_chunks"}
}
```

Uses: explaining a version to a reviewer; `kb-server reprepare --stale` (recorded settings
differ from the current ones); finding versions to re-embed after an embedder change. A
version with `preparation` but no `index` isn't searchable with its current preparation.
Who ran what and when is in `indexing_jobs`, not here.

### Deleting (planned, after the worker)

```text
request  DELETE /documents/{id} (can_manage) → 202; status = deleting + a delete job, one
         transaction. Reads and search treat a deleting document as gone (a tombstone).
worker   1. remove its chunks from the index
         2. delete prepared/<document_id>/
         3. one transaction: delete the document row (cascades), then stored_files rows nothing uses
         4. delete those originals from the store
```

Steps 1–2 are safe to repeat; the database goes before the originals. The delete job's own row
cascades away, so the worker skips recording it. A crash between 3 and 4 leaves an orphaned
original, which `kb-server gc` removes later.

### Logging

Python `logging`, to stderr; Docker Compose or systemd collect it. Configured once at startup
(`KB_LOG_LEVEL`). Log lines carry ids, versions, profile names and codes only: never document
text, tokens or emails.

---

## 9. Authentication and authorization (built)

- Passwords: argon2id. Tokens: opaque (`kb_s_` session, `kb_k_` key), SHA-256 stored only,
  shown once, never logged.
- `Authorization: Bearer <token>` → `Principal(user, credential_id, credential_kind,
  acting_for_others)`.
- A front-end key acts for the user named in `KB_ACT_FOR_HEADERS`; the user must exist and be
  active. `Principal.is_admin` is false while acting for someone.

| Role | May |
|---|---|
| reader | search, answer, download — within their groups |
| manager | + documents, batches, relations, retirements — in collections they manage |
| admin | everything; users, groups, keys, collections, reindex |

---

## 10. REST API: `/api/v1`

```text
errors       {"error": {"code": "document_not_found", "message": "…", "details": …}} + HTTP status
lists        ?limit=50&cursor=…  →  {"items": [...], "next_cursor": "…" | null}
long work    202 Accepted + job body + Location: /api/v1/jobs/{id}
```

| Area | Endpoints |
|---|---|
| system (built) | `GET /health`  `GET /ready` |
| auth (built) | `POST /auth/login`  `POST /auth/logout`  `GET /me`  `POST /me/password` |
| keys (built) | `POST /keys`  `GET /keys`  `DELETE /keys/{id}`  admin variants |
| users (built) | `POST /users`  `GET /users`  `GET /users/{id}`  `PATCH /users/{id}`  … |
| groups (built) | `POST /groups`  `GET /groups`  `DELETE /groups/{id}`  members |
| collections (built) | CRUD, `PUT /collections/{id}/groups`, managers |
| documents | `POST /documents` (built)  `GET /documents/{id}` (built)  `GET /documents`  `PATCH /documents/{id}`  `PUT /documents/{id}/file`  `DELETE /documents/{id}`  `GET /documents/{id}/versions`  `GET /documents/{id}/file`  `GET /documents/{id}/review` |
| jobs | `GET /jobs`  `GET /jobs/{id}` |
| search | `POST /search`  `POST /answer` |
| later | batches, relations, retirements, feedback, `/admin/*`, `/openai/v1` |

Error codes so far: 401 `login_failed`, `not_authenticated`; 403 `not_allowed`; 404
`user_not_found`, `group_not_found`, `collection_not_found`, `document_not_found`,
`credential_not_found`; 409 `email_taken`, `group_name_taken`, `group_in_use`,
`collection_slug_taken`, `duplicate_file`; 413 `file_too_large`; 415 `unsupported_file_type`;
422 `invalid_request`.

## 11. OpenAI-compatible: `/openai/v1` (later)

```text
GET  /openai/v1/models              one model per collection + "knowledge-base" (all)
POST /openai/v1/chat/completions    conversation in, answer + sources out; stream supported
```

## 12. CLI (`kb-cli`, later)

HTTP only, config in `~/.config/kb/config.toml`. `kb login | docs | jobs | search | ask | keys
| users | groups | collections`.

## 13. Server commands (`kb-server`)

```text
kb-server create-admin              first admin user (built)
kb-server serve [--reload]          uvicorn (built)
kb-server worker                    run jobs
kb-server migrate                   alembic upgrade head
kb-server reprepare --stale|--all   queue prepare jobs for out-of-date versions
kb-server gc                        remove originals no row needs
```

---

## 14. Build order

1. Skeleton: settings, database, wiring, app factory, health, Alembic, tests. **(built)**
2. Users, groups, credentials, login, API keys, `Principal`, `create-admin`. **(built)**
3. Collections. **(built)**
4. Documents and the worker:
   - a. FileStore, document tables, upload, `GET /documents/{id}`. **(built)**
   - b. The worker:
     1. `prepared.py` and `PreparationSettings` **(built)**
     2. the `plain` profile **(built)**
     3. `ProfileChooser`, `ClaimedJob`, `PermanentJobError` **(built)**
     4. the prepare job **(built)**
     5. the index job **(next)**
     6. `Worker`: claim → load → superseded check → handler → record; retries; the stuck-job
        sweep; polling; graceful shutdown
     7. settings, `IndexTools`, the `kb-server[worker]` extra, `kb-server worker`, Qdrant in
        Docker Compose, logging setup
     8. a real end-to-end run with a real regulation and bge-m3
     9. deleting documents
5. Search: `POST /search` with the access filter. Moved ahead of the read side because it is
   the reason the system exists, and it tests the index decisions early.
6. Read side: list with paging, versions, download, review page, `GET /jobs/{id}`.
7. Editing: title and groups (`sync_attributes`), new versions, outline preview, profile
   override.
8. Operations: `reprepare`, re-index after an embedder change, `gc`, backups.
9. Retrieval evaluation: a labelled question set under `data/`, hit rate and MRR.
10. Answers, the OpenAI-compatible surface, batches, relations and retirements, feedback, CLI.

Project hygiene alongside: CI (ruff, mypy, pytest with Postgres), a pre-commit guard refusing
`data/`, `.docx`, `.pdf`, `.env`, and short decision records.

## 15. Decisions that are easy to undo by accident

| Decision | Why |
|---|---|
| Hidden things are 404, not 403 | A 403 confirms that something exists |
| Repositories report, services raise | One place decides what's an error |
| No `profile` column on `documents` | Chosen automatically; overrides have their own table |
| Prepared output per version, no cache | It retires with its document; nothing stale is reused |
| `save_prepared` deletes before writing | A half-save loads as nothing, never as a mix |
| Handlers never touch the database | They can be tested with a file store only |
| Search filters inside the index | Post-filtering loses results and leaks hidden hits |
| Provenance holds inputs and decisions, not results | Results can be rebuilt; settings can't |
| One worker | No races between jobs; follow-up jobs run in order |
| `vn_regulation` isn't the default | Most documents aren't regulations; `plain` is the safe fallback |

## 16. Later

- Framework: outline levels on physical blocks; generic profiles (`headings`, `numbered`);
  profile scoring; extract the physical document once for several profiles.
- Answer tone per user; routing of non-questions; section scoping by heading relevance.
- Unresolved references to documents not yet uploaded.
- Conversations stored server-side; nested collections; automatic related documents.
- An audit trail; deployment; index rebuild tools.
