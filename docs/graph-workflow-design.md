# Retrieval as a graph of stages

Status: design discussion, nothing built. Branch: `features/retrieval-redesign`.

The project is named **Salemdex**. This engine is its first package under that name,
`salemdex-retrieval` (section 10).

## 1. The problem

The project is a knowledge base: a product whose answers must be **accurate** and **safe**.
Every answer is built from what search returns, so search is where both are won or lost:

- An answer can't be better than its evidence, so search caps accuracy.
- An answer can only reveal what search gave it. If search returns only what the asker may see,
  everything built on top is safe without checking again.
- Evidence must be checkable: each passage is the source's own words, traced to its source.

Retrieval patterns range from one vector search to agentic loops. Hard-wiring one of them means
rewriting search for the next. The current framework API also has these problems:

- A filter can't OR across fields ("my groups OR my managed collections").
- There are two filter mechanisms: fixed fields (`document_ids`, `chunk_keys`, `reference_keys`)
  and `attributes`.
- `search_chunks` takes 11 parameters, mixing tools and settings. `limit` and `rerank_depth` are
  coupled through `max()`.
- `ChunkFilter()` means everything. Removing a document is `replace_document(id, [], [])`.
- Ids are `str`, which already caused a UUID bug.
- Reading the index needs the private `_payload` module and an embedder identity.
- Indexed attributes are declared in two places (`QDRANT_INDEXED_ATTRIBUTES` and the chunks).

## 2. Search in its most basic form

```text
search(question, asker) → passages, best first
passage = the source's own words + where they come from (document, version, location) + rank
```

| # | Property | Kind |
|---|---|---|
| 1 | **Allowed:** only what the asker may see; hidden things leave no trace, not even in counts | guarantee |
| 2 | **Current:** nothing deleted; one whole version per document, never a mix | guarantee |
| 3 | **Faithful:** the text is exactly the source's, traceable to document, version, location | guarantee |
| 4 | **Ranked:** what most likely answers the question comes first | quality |

**Guarantees** always hold, are tested as yes/no, and are never traded for quality. **Quality**
is measured (hit rate, MRR) and tuned. This split drives the whole design.

Conceptually there are two steps: **narrow** (what this asker may see, now) and **order**
(relevance, keep top k). Everything else refines one of the two, or sits above search.

## 3. The basic pipeline

```text
request: question, asker, scope?, k
│
├─ 1. Encode         question                      → query vectors (dense, sparse…)
├─ 2. Allow          asker, scope?                 → filter
├─ 3. Retrieve       query vectors, filter, depth  → candidates: chunk + index score
├─ 4. Confirm        candidates                    → candidates still current
├─ 5. Rerank         question, candidate texts     → candidates reordered + rerank score
└─ 6. Cut & shape    candidates, k                 → passages: text, source, score
```

- 1 and 2 are independent.
- 4 runs before 5: no reranking of candidates about to be dropped.
- The filter runs inside the index. Why: filtering afterwards loses results and reveals hidden
  hits. The index applies access; the database confirms currency.
- **One score per passage:** from the last stage that ordered the candidates (rerank if present,
  else the index or fusion score). Comparable only within one response. Why: scores from
  different stages, or different questions, aren't on the same scale.

## 4. Fixed frame, pluggable stages

Only the quality stages are pluggable. Why: if every stage can be swapped, a plugin can break a
guarantee (e.g. an "expand to related chunks" stage that pulls in chunks the asker can't see).

```text
fixed frame (always runs, can't be removed)     pluggable inside it
─────────────────────────────────────────────    ──────────────────────────
allow    → every candidate source gets the filter  query transforms
confirm  → current check on whatever comes out     sources, merging
faithful → passages carry source text + origin     reranking, expansion, pruning
```

### Stage kinds

| Kind | In → out | Examples |
|---|---|---|
| query transform | question → queries | rewrite, multi-query, HyDE, decomposition |
| source | query, filter → candidates | dense, sparse, keyword, generated entries |
| merge | candidate lists → candidates | RRF, union |
| expand | candidates, filter → candidates | neighbours, parent section, linked chunks |
| rerank | question, candidates → candidates | cross-encoder, LLM judge |
| prune | candidates → candidates | dedupe, threshold, diversity |

Every stage that adds candidates (source, expand) receives the filter. The framework fixes
*where* the guarantees apply. The app decides *what* they mean (section 6).

## 5. Statements and the executor (the SQLAlchemy pattern)

**A statement describes a search; an executor runs it**, like `select()` and
`session.execute()`.

```text
stmt = search(question).retrieve(dense, sparse).merge(rrf).rerank(cross_encoder).take(10)
passages = engine.session(access=access).execute(stmt)
```

- **Access is bound to the session, not the statement.** Why: a statement can't forget the
  access filter. The executor adds it to every source and puts the currency gate at the end.
  (How the slots plug in: section 6.)
- **Type-state builder:** each method returns a type that offers only the steps allowed next, so
  mypy checks the order:

  ```text
  Question   → .rewrite() .split()                  → Queries
  Queries    → .retrieve(sources…)                  → Candidates
  Candidates → .merge() .rerank() .expand() .prune() → Candidates
  Candidates → .take(k)                             → Passages   (terminal)
  ```

- **Immutable and generative:** each method returns a new statement. Statements are data: they
  can be inspected, logged as provenance and replayed.
- **Composite stages:** a stage can contain stages (a merge over several sources). That gives a
  tree, which covers fan-out and fan-in. There is no free-form graph of untyped nodes. Why: it
  would lose strict typing and make the guarantees hard to prove.

### Candidates: what flows between stages

A **passage** is what comes out at the end. A **candidate** is what flows between stages. Later
stages can only use what earlier stages kept, so a candidate keeps its history:

```text
Candidate
  chunk   the chunk (id, text, column values)
  score   the current score: the last trail entry's
  trail   one entry per stage that touched it:
          stage, score or rank, found through (which source, or which generated entry)
```

- Why a trail: it explains a hit ("why is this here?"), lets each stage be evaluated on its
  own, gives answers their provenance, and lets a prune use a specific stage's score ("rerank
  score ≥ x").
- "Found through" generalises today's `found_through` to every stage.
- The trail holds ids, scores and stage names, never text. Why: it can then be logged as
  provenance.
- The passage's single score is the last trail entry's (section 3).
- Rejected: only chunk + score (each stage overwrites the score, so history is lost); ids only
  with text loaded when needed (more complex loading, not needed at depth 50).

## 6. Data: generic chunks, a schema, fixed slots

**Flexible data, restricted structure.** The framework knows structure: what can follow what.
The app decides meaning: what a document, an access group or a reference is. A chunk has no
built-in concept of document, access or reference.

### The chunk

```text
Chunk
  id             UUID, unique across the whole index
  text           what is shown and returned; never changed
  embedded_text  None → text is embedded
  values         {column: value}, checked against the schema
```

- **The id is unique across the index**, not only within a document. The app derives it
  deterministically, e.g. `uuid5(document_id, version, position)`. Why: Qdrant takes UUIDs
  natively, mypy can check the type, and re-indexing the same version gives the same ids, so a
  retried write is idempotent. The version is part of the id so that a new version's chunks
  never overwrite the old version's while both are in the index (see "Currency" below).
- **`embedded_text` is where context goes**, e.g. the headings path prefixed to the text. That
  is the reason it can differ from `text`.
- **Vectors aren't part of the chunk.** Vector names (dense, sparse) and the embedder belong to
  the index's configuration.
- **Everything else is a column:** document id, version, position, headings, reference keys,
  access groups.
- **"Faithful" narrows.** The framework guarantees only that `text` comes back exactly as it
  was indexed. Where it came from is the app's columns.

### The schema

Like SQLAlchemy's `Table` / `Column`: columns are declared once, with their types and whether
they are indexed. Each column is a typed handle on one chunk value. Its type `T` is checked
wherever the column is used.

```python
class KbChunks(Schema):
    document_id = One(UUID, indexed=True)          # One[UUID]
    version     = One(int, indexed=True)           # One[int]
    position    = One(int)
    retired_on  = One(datetime, required=False)    # One[datetime | None]
    headings    = Many(str)                        # Many[str]: several values
    groups      = Many(UUID, indexed=True)
```

**Two column kinds: `One[T]` (a single value) and `Many[T]` (several values).** Why not a
single `Column[T]`: mypy can't infer `T` from `tuple[str, ...]`, only from a real class. Also,
the operations differ: `eq` and ranges fit one value, `any_of` / `all_of` fit several.

- `T` is one of `str`, `int`, `float`, `bool`, `UUID`, `datetime`: what every backend can store.
- `required=False` makes a column optional: `One(UUID, required=False)` is `One[UUID | None]`.

**What `T` buys at each use:**

| Use | Example | What mypy catches |
|---|---|---|
| filter | `KbChunks.document_id.eq(doc_id)` | `eq("abc")`: a `str` where a `UUID` is expected |
| ordered filters | `KbChunks.version.ge(2)` | `ge` on a `UUID` column. It exists only for int, float, datetime (a self-type on the method) |
| multi-valued | `KbChunks.groups.any_of(ids)` | `eq` on a `Many`, `any_of` on a `One` |
| read | `chunk.get(KbChunks.document_id)` → `UUID` | no `str` or `Any` coming back. It's generic: `get(col: One[T]) -> T` |
| stage settings | `expand(order_by=KbChunks.position)` | `order_by` takes only ordered `One` columns |

**Writing chunks: typed columns plus a value builder.**

```python
Chunk(id=…, text=…, values=[KbChunks.document_id.value(doc_id), KbChunks.version.value(3), …])
```

- Every column-value pair is type-checked. A missing required column is caught at run time, when
  the index checks the write against its schema.
- Why this level: chunks are built in one place (the index handler), while filters and reads are
  spread everywhere, and those are fully typed.
- Possible later: the SQLAlchemy 2.0 style. The schema class is also a typed record
  (`dataclass_transform` plus descriptors), so mypy catches missing fields too. It needs a small
  experiment to see how well mypy handles it. Filters and reads wouldn't change.
- Rejected: a plain dataclass with `Annotated` metadata. Writes would be typed, but filters
  would name columns by string and lose their typing.

**Each index owns its schema; the engine reads it from the index (`index.schema`).**

- Why: one source of truth. Two copies, one on the index and one on the engine, could drift.
  The compile step would then approve a filter on a column the index doesn't have.
- There is more than one schema (chunks, generated entries), each with its own index.
- The compile step checks that every column in a statement belongs to the schema of the index it
  runs against. A chunk column can't filter the entries index.
- **The index checks its stored data when it opens:** it compares the schema with the backend
  (e.g. Qdrant payload indexes), then creates what's missing or fails with a clear message.
  Why: otherwise filters run slowly or wrongly without saying so.
- Not now: statements generic over their schema (`Candidates[KbChunks]`), which would let mypy
  catch cross-schema columns. The compile check is enough to start with.

**Other uses of the schema:**

- Backend indexes follow from the declaration. That replaces the duplicated
  `QDRANT_INDEXED_ATTRIBUTES`.
- Values are encoded to the backend's format and decoded back (`UUID`, `datetime`). Why: that
  removes the need for the private `_payload` module.
- **Writes are filter-based**, like SQL's `DELETE … WHERE`: `index.add(chunks)`,
  `index.delete(where=…)`, `index.replace(where=KbChunks.document_id.eq(x), with_=new_chunks)`.
  The framework needs no document concept even for index upkeep, and
  `replace_document(id, [], [])` disappears.
- **Stages that need meaning take columns as settings:**
  `expand(neighbours, order_by=KbChunks.position, within=KbChunks.document_id)`,
  `prune(dedupe, by=KbChunks.document_id)`. The compile step checks that those columns exist
  and have the right types.
- A **standard schema preset** (document id, position, headings) is offered as a convenience,
  never as a requirement. Why: without one, every app would rebuild the basics.

### Filter expressions

- **AND / OR / NOT over schema columns and chunk ids.** One mechanism for every filter. Why:
  today's fixed fields plus `attributes` can't OR across fields.
- **Methods are the API:** `KbChunks.document_id.eq(x)`, `any_of(…)`. Why: they read plainly and
  type-check under strict mypy. Operator overloading (`==`, `|`) may come later as sugar over
  the methods, only if they prove ugly.
- **Each backend compiles the expression tree:** Qdrant into must / should / must_not, the
  in-memory index into a predicate. A construct the backend can't run fails at compile time,
  before any work.

### Generated entries

An entry's text isn't the source's words, so it can't be a chunk's `text`. Entries get their
own schema with a column that points to chunk ids. A source stage searches the entries and
returns the chunks they lead to, recording "found through" in each candidate's history. They
become a separate source, not a special case inside the index.

### The guarantee slots

The executor can't enforce access or currency on data it doesn't understand. So the frame has
fixed **slots**, and the app fills them:

```text
framework (generic)                         app (server) fills
─────────────────────────────────────────   ─────────────────────────────────────
schema: declared columns, types, indexes  ← document_id, version, groups, collection…
chunk = id + text + column values
filter expressions over columns
access slot:   a filter, given per session ← "shares a group OR manages the collection"
validity slot: candidates → kept ones      ← "document active, last indexed version"
stages, executor, plan, gates
```

- **Every slot must be filled.** An engine can't be built without an access policy. An app that
  wants no access control says so with an explicit `AllowEverything`, so the absence is visible.
- The executor still decides *where* the slots apply: the access filter goes into every source
  and expand stage, and the validity check runs before rerank and again at the final gate.
- **What access means is the app's design, not the framework's.** The framework only requires
  the access slot to be a filter expression. How the app builds it (which columns,
  combinators or a denormalised column) is decided in the app.

### How the slots plug in

**Validity: a port the engine holds.**

```text
engine = Engine(index=…, validity=ServerValidity(session_factory))   # Protocol: candidates → kept
```

- It is part of the frame, not a stage the app writes into statements. Why: a statement author
  or an agent could leave a stage out, and stale results would pass.
- It doesn't depend on the asker ("is the document active, is this its indexed version?"), so
  it belongs to the long-lived engine. The executor runs it before rerank and at the final gate.

**Access: the filter is passed when a session opens.**

```text
access = documents.access_filter(principal)     # server service, with its own DB session
session = engine.session(access=access)
```

- Why this over a policy the engine holds (asker → filter): it is simpler. The framework has no
  asker type and no per-request plumbing. In the server, building the filter needs a DB lookup
  (the principal doesn't carry its groups or managed collections), and services already check
  permissions with a DB session.
- The cost: the guarantee rests on a convention, not on structure. A call site that passes a
  filter that's too broad breaks it. Kept robust by:
  - **one server service opens sessions**; `/search`, `/answer` and agents all go through it;
  - a test of the filter that service builds for each kind of asker;
  - an import-boundary check in CI: only that module imports `Engine`.
- **When to switch to an engine-held policy:** when agents or other entry points need to open
  sessions themselves. The engine sees a filter either way, so the switch stays cheap.

### Currency: show the last indexed version

How the server fills the validity slot. While a new version waits to be indexed, search keeps
showing the last indexed version. Why: hiding the document would make it disappear during every
re-index (minutes on CPU), and for good if the new version fails to index.

1. **The database records which version is in the index:** `documents.indexed_version` (one
   migration). The worker sets it in the transaction that marks the index job done. The validity
   check keeps a hit when the document is active and the hit's version is `indexed_version`, not
   `current_version`.
2. **Chunk ids include the version** (section 6, "The chunk"), so v2's chunks never overwrite
   v1's.
3. **The write is split around the commit.** This is the pointer-flip pattern: write the new
   copy, switch one pointer, remove the old copy.

   ```text
   handler   index.add(v2 chunks)                     v1 still served: indexed_version = 1
   worker    commit indexed_version = 2                the switch: one row, atomic
   after     index.delete(where=document is x AND version isn't 2)
   ```

   - From a reader's view the switch is atomic: all of v1, then all of v2, never a mix and never
     nothing.
   - A crash after the commit leaves v1's chunks behind. The validity check already hides them,
     and `gc` removes them. It's the same idea as "originals are deleted after the commit".

Consequences:

- **Access follows the document, not the version.** Old chunks stay in the index longer. A change
  to the document's groups must update every chunk of the document, whichever version
  (`where=KbChunks.document_id.eq(x)`). Otherwise the old version would be served with stale
  groups: a leak.
- **A failed new version keeps the old one served indefinitely.** That's acceptable, but it must
  not be silent:
  - every passage carries its version, so the API can mark a hit "a newer version is being
    processed" (`current_version > indexed_version`);
  - the failure is visible on the document (operations, build step 8).

## 7. Validation

| When | What | Why |
|---|---|---|
| Before any work ("compile") | stage order, the index supports the filter, the tools exist, limits fit (k ≤ depth) | nothing is spent on a statement that would fail |
| During the run | empty results, wrong score counts, a tool failing | only known as it runs |

mypy covers statements written in code. Statements built at run time (from config, or by an
agent) still need the compile check, so the executor always runs it.

## 8. Loops

Loops are common in advanced retrieval (retrieve → grade → rewrite → retrieve again: CRAG,
Self-RAG, iterative retrieval). **Statements stay acyclic. A controller above the executor owns
the loop.**

```text
controller (the loop)       run → look at the candidates → next statement, or stop
  └─ statement (acyclic)    validated as a whole, finite, the same gates every round
```

- Every round stays checkable. A cyclic graph can't be fully validated up front.
- Termination is explicit: the controller owns the limits (most rounds, time or cost budget).
- The gates hold every round. The controller re-checks the final output, because a document can
  be deleted mid-loop.
- Possible later: a declarative `repeat(body, until=…, max_rounds=…)` that compiles to the
  controller. Its body is still an acyclic statement.

An agent step continues from earlier results, e.g. `from_candidates(previous).expand(...)`.
These still go through the same executor.

## 9. Agents that recommend decisions

Retrieval is the agent's evidence layer. The guarantees carry up to the recommendation unchanged.

```text
decision agent (controller)
  1. frame     the decision → what must be known (options, constraints, criteria)
  2. gather    one statement per need  ──→ a session with the user's access
  3. weigh     options against evidence (the LLM's work)
  4. check     is every claim backed by a passage? does any evidence conflict?
  5. answer    recommendation + cited evidence, or "not enough evidence"
```

| Concern | Provided by |
|---|---|
| No recommendation rests on, or hints at, hidden knowledge | every statement runs in a session with the user's access |
| No deciding on outdated knowledge | the currency gate on every statement |
| "Why did it recommend this?" | passages with origin; statements logged and replayable |
| Coverage: every constraint that applies, counter-evidence | different statements per need ("supports A", "argues against A", "everything in scope") |
| Knowing when not to recommend | scores and gates, so the agent can abstain |
| Is it improving? | fixed statements per step, evaluated one step at a time |

The framework doesn't make step 3 correct. It guarantees the reasoning starts from permitted,
current, traceable evidence. Step 4 checks the output against that evidence, and that check is
where most of the remaining safety comes from.

## 10. Scope of the first version

Build only the stage kinds that basic search needs: **retrieve, merge, rerank, take**, with the
filter expression and the executor's frame. Add rewrite, expand, prune and the controller when
something uses them. Why: the type-state structure is the costly part to get right; adding
stage kinds afterwards is cheap.

**Package:** the engine is a new uv workspace member in this repo, the distribution
`salemdex-retrieval`.

- It never imports `document_processor`. An import-boundary check in CI enforces it. Why: the
  engine is generic; turning `chunking.Chunk` into an index chunk (today's `ingest` and
  `address_keys`) is app code, in the server, which declares the schema anyway.
- Its extras hold the backends and models: `qdrant`, `huggingface` (embedder, reranker).
- A separate repo waits until the engine needs its own releases.
- The family follows later, one rename per commit: `salemdex-ingest` (today's
  `document_processor`), `salemdex-server` (today's `kb_server`), and the repo itself.

**Migration:** build the new API alongside the old one, port the callers (server indexing
handlers, search, evaluation, tests), then delete the old API.

**Rerank depth is a per-stage setting** (`rerank(name, depth=…)`), never hard-coded. Its default
is chosen later, by measurement: latency per depth on the target hardware, and hit rate / MRR per
depth on the evaluation set (build step 9). Pick the smallest depth where quality stops
improving, within the waiting-time budget. Why it matters: a cross-encoder scores each
question–passage pair through a large model, so depth trades recall (it can only lift what it
sees) against waiting time and throughput, and `limit` ≤ depth caps k. For the measurement, the
executor records each stage's duration in the session's trace.

## 11. Open questions

- **The import name:** `salemdex_retrieval`, or a namespace package `salemdex.retrieval` shared by
  the family? And the folder: `packages/retrieval/`?
- **The standard schema preset:** which columns it holds.
- **The agent step API** (later, with the controller):
  - How a statement starts from earlier candidates. A likely shape: the LLM sees short handles
    (`p2`), and the controller maps them back; an invented handle doesn't resolve. Earlier
    candidates pass the filter and the validity check again, since documents change between
    rounds.
  - What the controller, and so the LLM, sees after a tool call: passage text and handles, how
    much of the trail, how many passages per round (the context window limits this).
  - Everything the LLM writes is untrusted input. The session's access is what keeps that safe.
