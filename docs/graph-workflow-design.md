# Retrieval as a graph of stages

Status: build step 1 done; step 2 next (section 10). Branch: `features/retrieval-redesign`.

- Built, in `packages/retrieval` (`salemdex_retrieval`), with tests:
  - `schema.py`: `One` / `Many` fields, `Schema` with presets, `OneValue` / `ManyValues`,
    the value checks;
  - `filters.py`: `Equals` (`eq`), `AnyOf` (`any_of`), and `and_` / `or_` / `not_`;
  - `chunks.py`: `Chunk`, and `Chunk.check_fits(schema)` raising `SchemaMismatchError`;
  - `index/in_memory.py`: `InMemoryIndex` with `put`, `find`, `delete` and `replace`.
- Next: build step 2, the Qdrant backend, matching the in-memory index's behaviour, and the
  `Index` protocol the two share.
- Later, when a caller needs it: `chunk.get(field)` for typed reads (step 3 or 4), `in_` on
  `One`, filtering by chunk ids.

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
  chunk   the chunk (id, text, field values)
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
  embedded_text  what the model embeds; the text itself unless one is given
  field_values   {field: value}, keyed by the field object; read-only
```

- **Built with keywords only.** Why: `text` and `embedded_text` are both strings, and swapped
  by position, search would return the context as the source's words.
- **`embedded_text` is optional to give, always a string to read.** Left out, the constructor
  fills in the text. Why: no reader handles `None`, and nobody has to know what it means.
- **Text stays out of `repr`** (`field(repr=False)`). Why: a chunk in a log line or an error
  would otherwise print restricted text.
- **Values are keyed by the `Field` object, not its name.** Why: `KbChunks.document_id` and
  `Entries.document_id` share a name but are different fields; only identity tells them
  apart. It commits `Field` to never defining `__eq__` (so no `==` filter sugar): a dict
  would compare keys with it.
- **Two values with one name are refused when the chunk is built** (`ValueError`): the same
  field twice, or fields from two schemas. No schema has two fields with one name, so such a
  chunk can't fit anywhere.
- **`chunk.check_fits(schema)`** checks a value for every field of the schema and no other,
  comparing fields as objects; an empty `Many` counts as a value. It raises one
  `SchemaMismatchError` (a `ValueError`) naming the chunk id and every extra and missing
  field. Why a separate error from `SchemaError`: the schema is fine, the data doesn't fit it,
  so a caller (the worker) can treat it as permanent.

- **The id is unique across the index**, not only within a document. The app derives it
  deterministically, e.g. `uuid5(document_id, version, position)`. Why: Qdrant takes UUIDs
  natively, mypy can check the type, and re-indexing the same version gives the same ids, so a
  retried write is idempotent. The version is part of the id so that a new version's chunks
  never overwrite the old version's while both are in the index (see "Currency" below).
- **`embedded_text` is where context goes**, e.g. the headings path prefixed to the text. That
  is the reason it can differ from `text`.
- **Vectors aren't part of the chunk.** Vector names (dense, sparse) and the embedder belong to
  the index's configuration.
- **Everything else is a field:** document id, version, position, headings, reference keys,
  access groups.
- **"Faithful" narrows.** The framework guarantees only that `text` comes back exactly as it
  was indexed. Where it came from is the app's fields.

### The schema

Declared like SQLAlchemy's `Table` / `Column`, but called fields: a chunk isn't a table row.
Fields are declared once, with their types and whether they are indexed. Each field is a
typed handle on one chunk value. Its type `T` is checked wherever the field is used.

```python
class KbChunks(Schema):
    document_id = One(UUID, indexed=True)  # One[UUID]
    version = One(int, indexed=True)  # One[int]
    position = One(int)
    headings = Many(str)  # Many[str]: several values, maybe none
    groups = Many(UUID, indexed=True)
```

**Two field kinds: `One[T]` (a single value) and `Many[T]` (several values).** Why not a
single `Field[T]`: mypy can't infer `T` from `tuple[str, ...]`, only from a real class. Also,
the operations differ: `eq` and ranges fit one value, `any_of` / `all_of` fit several.

- `T` is one of `str`, `int`, `float`, `bool`, `UUID`, `datetime`: what every backend can store.
  `Field.__init__` checks this and names the allowed types. It compares classes exactly, not
  with `issubclass`, because `bool` is a subclass of `int`.
- **Values are checked wherever they enter:** in `value()` (writes) and in `eq` / `any_of`
  (filters), with one shared check. Why filters too: statements built at run time (config, an
  agent) arrive as `Any`, and a `str` where a `UUID` belongs would match nothing instead of
  failing.
  - The type must match exactly; an `int` is widened to `float` for a `float` field.
  - A `datetime` must have a timezone.
  - A `float` must be finite, and an `int` must fit in 64 bits (`ValueError`). Why: JSON has
    no NaN or infinity, Qdrant stores int64, and `nan` never equals itself, so `eq(nan)` would
    match nothing without saying so.
  - `Many.value` and `any_of` refuse a `str`, `bytes` or `Mapping`: each is iterable, and
    would be taken apart into characters, ints or keys.
- `One(UUID)` is the call; `One[UUID]` is the type mypy infers from it. The type is passed as an
  argument because the run time needs it too (checking and encoding values, choosing backend
  index types), and generic parameters are erased at run time.

**Every field is required; there are no optional fields (for now).**

- Every chunk has a value for every field. A `Many` field may hold no values (`[]`), and that
  is how "none" is said for multi-valued data: no tags, no groups.
- Why: "missing" would need rules in every filter, like SQL's `NULL`, and both backends would
  have to apply them the same way. A `not_(…)` filter lets chunks without the field through (in
  Qdrant, `must_not` matches them), so a missing value could fail open. It also keeps `T` free
  of `None`, and `One` needs no overloads.
- **A missing or `null` value is a data error, caught at three points:**

  | Where | What happens |
  |---|---|
  | writes | every chunk is checked against the schema before it's sent; a missing value raises, naming the chunk id and field. An empty `Many` is written as `[]`, never left out |
  | the index opening | for each field, count the chunks where it's missing or `null` (in Qdrant, `is_empty` / `is_null` counts); if any exist, refuse to start: "412 chunks lack `version`; backfill or re-index" |
  | reads | a hit that still lacks a value raises a typed error naming the chunk id and field. No half-filled chunk, no guessed default |

  Why refuse to start: filters then only ever run over chunks that keep the promise. Why raise
  on reads rather than skip: skipping also fails safe, but data would vanish from results without
  anyone noticing.
- **No sentinel values** (`0`, `""`, an epoch date) to stand in for "missing". They quietly take
  part in filters: `page.lt(5)` would match every chunk whose page is the `0` sentinel. A field
  that would need one is the signal that optional fields are due.
- **When optional fields come:** with the first genuinely sparse, kind-specific field (a page,
  a sender, a sheet name). Adding them later breaks no existing declaration. Two rules come with
  them, so "missing" never matches by accident:
  - fields used by the guarantee slots (access, validity) must be required; the compile step
    checks it;
  - `not_` over an optional field is rejected unless the filter says what happens to missing
    values (`is_missing()` / `is_present()`).

**What `T` buys at each use:**

| Use | Example | What mypy catches |
|---|---|---|
| filter | `KbChunks.document_id.eq(doc_id)` | `eq("abc")`: a `str` where a `UUID` is expected |
| ordered filters | `KbChunks.version.ge(2)` | `ge` on a `UUID` field. It exists only for int, float, datetime (a self-type on the method) |
| multi-valued | `KbChunks.groups.any_of(ids)` | `eq` on a `Many`, `any_of` on a `One` |
| read | `chunk.get(KbChunks.document_id)` → `UUID` | no `str` or `Any` coming back. It's generic: `get(col: One[T]) -> T` |
| stage settings | `expand(order_by=KbChunks.position)` | `order_by` takes only ordered `One` fields |

**Writing chunks: typed fields plus a value builder.**

```python
Chunk(
    id=…,
    text=…,
    field_values=[KbChunks.document_id.value(doc_id), KbChunks.version.value(3), …],
)
```

- Every field-value pair is type-checked. A missing field is caught at run time, when the
  index calls `chunk.check_fits(schema)` on the write.
- **Two value classes, like the two field kinds:** `One.value` returns `OneValue[T]` (`value:
  T`) and `Many.value` returns `ManyValues[T]` (`values: tuple[T, ...]`).
  `type FieldValue = OneValue[Any] | ManyValues[Any]` names either, and a backend matches on
  the class to tell one value from several. Filters follow the same pattern: `Equals[T]`,
  `AnyOf[T]`, and `type Filter = Equals[Any] | AnyOf[Any]`.
  - Why not one generic `FieldValue[T]`: with `value: T`, a `Many` field's tuple isn't a `T`,
    and `value: T | tuple[T, ...]` wouldn't tie the shape to the kind of field.
  - Why the unions aren't generic: a chunk's values, and a filter combining fields, span
    several types, so every user would write `[Any]`. `T` is kept where it is known: `value()`,
    `eq` and `any_of` return the specific class.
- **Fields come only from the class body and from `Schema` bases.** A `Field` in a plain mixin
  isn't part of the schema; shared fields belong in a `Schema` preset.
- Why this level: chunks are built in one place (the index handler), while filters and reads are
  spread everywhere, and those are fully typed.
- Possible later: the SQLAlchemy 2.0 style. The schema class is also a typed record
  (`dataclass_transform` plus descriptors), so mypy catches missing fields too. It needs a small
  experiment to see how well mypy handles it. Filters and reads wouldn't change.
- Rejected: a plain dataclass with `Annotated` metadata. Writes would be typed, but filters
  would name fields by string and lose their typing.

**Each index owns its schema; the engine reads it from the index (`index.schema`).**

- Why: one source of truth. Two copies, one on the index and one on the engine, could drift.
  The compile step would then approve a filter on a field the index doesn't have.
- There is more than one schema (chunks, generated entries), each with its own index.
- **One index, one schema.** Why: `check_fits`, the filter check and the startup check all
  compare against a single schema; with two, a filter on one schema would run over the other's
  chunks, and `not_` would turn "matches nothing" into "matches everything". Several kinds of
  records are several indexes, combined by a merge stage. Two logical indexes sharing one Qdrant
  collection is a backend detail; each still has one schema. A schema that changes over time is
  a migration (backfill or re-index), not a second schema.
- The compile step checks that every field in a statement belongs to the schema of the index it
  runs against. A chunk field can't filter the entries index.
- **The index checks its stored data when it opens:** it compares the schema with the backend
  (e.g. Qdrant payload indexes), then creates what's missing or fails with a clear message.
  Why: otherwise filters run slowly or wrongly without saying so.
- Not now: statements generic over their schema (`Candidates[KbChunks]`), which would let mypy
  catch cross-schema fields. The compile check is enough to start with.

**Other uses of the schema:**

- Backend indexes follow from the declaration. That replaces the duplicated
  `QDRANT_INDEXED_ATTRIBUTES`.
- Values are encoded to the backend's format and decoded back (`UUID`, `datetime`). Why: that
  removes the need for the private `_payload` module.
- **Field values sit under their own payload key**, apart from the chunk's own parts:
  `{"text": …, "embedded_text": …, "fields": {"document_id": …, "groups": […]}}`. Only the
  Qdrant backend knows this layout; it turns `KbChunks.document_id` into the key
  `fields.document_id` for filters and payload indexes (Qdrant handles dotted keys). Why: in a
  flat payload, a field named `text` would overwrite the chunk's text and break "faithful"
  without an error. Reserving names instead would make every payload key the framework adds
  later a breaking change for existing schemas, and would put one backend's layout into the
  schema's rules.
- **Writes are filter-based**, like SQL's `DELETE … WHERE`: `index.put(chunks)`,
  `index.delete(where=…)`, `index.replace(where=KbChunks.document_id.eq(x), with_=new_chunks)`.
  The framework needs no document concept even for index upkeep, and
  `replace_document(id, [], [])` disappears.
  - **`put` is an upsert:** a stored chunk with the same id is overwritten. Why: a retried job
    writes the same deterministic ids again and must succeed (idempotent consumer); Qdrant's
    upsert works the same way. Named `put`, not `add`, because "add" reads as an insert and
    hides the overwrite.
  - **Every write is checked in full before anything is stored:** each chunk fits the schema,
    and no two share an id. Why: a batch failing halfway would leave part of a version beside
    the old one, the mix "Current" rules out.
  - **`replace` is SQL's `DELETE … WHERE` then `INSERT`:** the new chunks needn't match `where`
    ("replace v1 with v2" doesn't). Retrying it is safe because the ids are deterministic.
  - **Every field in a filter must belong to the index's schema, checked before any chunk is
    read or changed.** Why: `delete(where=not_(Entries.version.eq(2)))` on the chunks index
    would otherwise match nothing inside the `not_`, so everything, and empty the index.
  - **Writes are methods, not statements.** They come from the indexing worker, which has no
    asker, so nothing needs to sit between the caller and the index. Searches are statements
    (section 5) because the session must add the access filter to every one. Revisit if writes
    need a unit of work, a trace or a dry run.
  - **`index.find(where=…)` reads without any access filter**, like a raw database
    connection: only indexing code, tests and the engine may call it, never anything serving a
    user.
- **The in-memory index** (`InMemoryIndex`) is the reference behaviour every backend must match,
  for tests and single-process local runs. It is single-threaded and promises no more than
  Qdrant: no locks, so `replace` isn't made atomic there either (in Qdrant it is two requests).
  Why: the API and the worker are separate processes, so production needs Qdrant anyway, and a
  stronger in-memory index would let tests pass on guarantees Qdrant doesn't give. Vectors come
  with retrieve stages (build step 3).
- **Stages that need meaning take fields as settings:**
  `expand(neighbours, order_by=KbChunks.position, within=KbChunks.document_id)`,
  `prune(dedupe, by=KbChunks.document_id)`. The compile step checks that those fields exist
  and have the right types.
- A **standard schema preset** (document id, position, headings) is offered as a convenience,
  never as a requirement. Why: without one, every app would rebuild the basics.

### Filter expressions

- **AND / OR / NOT over schema fields and chunk ids.** One mechanism for every filter. Why:
  today's fixed fields plus `attributes` can't OR across fields.
- **Methods are the API:** `KbChunks.document_id.eq(x)`, `any_of(…)`. Why: they read plainly and
  type-check under strict mypy. Operator overloading (`==`, `|`) may come later as sugar over
  the methods, only if they prove ugly.
- **Combinators are functions, the nodes frozen dataclasses:** `and_(…)`, `or_(…)`, `not_(…)`
  build `And`, `Or`, `Not` (trailing underscores because `and`, `or`, `not` are keywords, as in
  SQLAlchemy). `type Filter = Equals[Any] | AnyOf[Any] | And | Or | Not` refers to itself.
  - **The checks live in the nodes' `__post_init__`**, so a node built directly is checked too.
    The functions only turn their arguments into a tuple.
  - **`and_` and `or_` need at least one filter.** Why: an empty AND matches everything, so
    `delete(where=and_(*conditions))` with an empty list would empty the index (the old
    `ChunkFilter()` problem). One filter is allowed: it's harmless, and lists of conditions
    vary in length.
  - **Each argument must be a filter**, checked at run time. Why: fields don't define `__eq__`,
    so `KbChunks.version == 2` is `False`, not an error; filters built at run time arrive as
    `Any`. The check reads its classes from the `Filter` alias, so a new node can't be missed.
- **Each backend compiles the expression tree:** Qdrant into must / should / must_not, the
  in-memory index into a predicate. A construct the backend can't run, or a field outside the
  index's schema, fails at compile time, before any work. The children are compiled first, so
  the whole tree is checked even where `or_` would stop early.
- Not yet: filtering by chunk ids (first caller: re-checking earlier candidates, step 3) and
  `in_` on `One` (first caller: scoping a search to collections or documents).

### Generated entries

An entry's text isn't the source's words, so it can't be a chunk's `text`. Entries get their
own schema with a field that points to chunk ids. A source stage searches the entries and
returns the chunks they lead to, recording "found through" in each candidate's history. They
become a separate source, not a special case inside the index.

### The guarantee slots

The executor can't enforce access or currency on data it doesn't understand. So the frame has
fixed **slots**, and the app fills them:

```text
framework (generic)                         app (server) fills
─────────────────────────────────────────   ─────────────────────────────────────
schema: declared fields, types, indexes  ← document_id, version, groups, collection…
chunk = id + text + field values
filter expressions over fields
access slot:   a filter, given per session ← "shares a group OR manages the collection"
validity slot: candidates → kept ones      ← "document active, last indexed version"
stages, executor, plan, gates
```

- **Every slot must be filled.** An engine can't be built without an access policy. An app that
  wants no access control says so with an explicit `AllowEverything`, so the absence is visible.
- The executor still decides *where* the slots apply: the access filter goes into every source
  and expand stage, and the validity check runs before rerank and again at the final gate.
- **What access means is the app's design, not the framework's.** The framework only requires
  the access slot to be a filter expression. How the app builds it (which fields,
  combinators or a denormalised field) is decided in the app.

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
   handler   index.put(v2 chunks)                     v1 still served: indexed_version = 1
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

**Package:** the engine is a new uv workspace member in this repo: the distribution
`salemdex-retrieval`, imported as `salemdex_retrieval`. Why not a namespace package
(`salemdex.retrieval`): it needs special care in hatch and mypy, and no member could have a
`salemdex/__init__.py`. Pydantic does the same with `pydantic_core`.

- **Folder:** `packages/retrieval/` at the repo root. Libraries go in `packages/`, deployables
  (server, CLI) in `apps/`.
- **Scope: storing and finding chunks.**

  | In `salemdex_retrieval` | Why here |
  |---|---|
  | schema, chunks, filter expressions | the data being retrieved and the language for selecting it |
  | index backends, reads and writes (`put`, `replace`, `delete`) | the index is what retrieval reads; writing it is the other half (as SQLAlchemy has `insert` beside `select`) |
  | embedders, rerankers | used by both writes and searches |
  | statements, stages, executor, session, gates | retrieval itself |

  | In a later package (e.g. `salemdex_agents`) | Why separate |
  |---|---|
  | the controller, agent loops, LLM tools | the controller sits above the executor (section 8); a package boundary makes that layering real |
  | answer generation, LLM providers | separate dependencies; retrieval stays usable without an LLM |

  Why the split also helps safety: an agents package can only use retrieval's public API, so
  agents must go through a session with the user's access. The package boundary enforces it.

- It never imports `document_processor`. An import-boundary check in CI enforces it. Why: the
  engine is generic; turning `chunking.Chunk` into an index chunk (today's `ingest` and
  `address_keys`) is app code, in the server, which declares the schema anyway.
- Its extras hold the backends and models: `qdrant`, `huggingface` (embedder, reranker).
- A separate repo waits until the engine needs its own releases.
- The family follows later, one rename per commit: `salemdex-ingest` (today's
  `document_processor`, moving to `packages/ingest/`) and `salemdex-server` (today's
  `kb_server`). The GitHub repo is already `salemdex`.

**Build order.** Each layer is tested before the next one depends on it.

1. **Schema, chunk, filter expressions**, plus the in-memory index that translates them. All of
   it is testable without Qdrant. **(done)**
2. **The Qdrant backend:** translating filters, creating payload indexes from the schema, the
   startup check, `put` / `find` / `replace` / `delete`, with the in-memory index's tests run
   against it too. Extract the `Index` protocol the two share. **(next)**
3. **Statements and the executor** with retrieve, merge, rerank and take, plus the session, the
   access filter and the validity check.
4. **Port the server's indexing handlers** to the new API, then build search on it.

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

- **The standard schema preset:** which fields it holds.
- **The agent step API** (later, with the controller):
  - How a statement starts from earlier candidates. A likely shape: the LLM sees short handles
    (`p2`), and the controller maps them back; an invented handle doesn't resolve. Earlier
    candidates pass the filter and the validity check again, since documents change between
    rounds.
  - What the controller, and so the LLM, sees after a tool call: passage text and handles, how
    much of the trail, how many passages per round (the context window limits this).
  - Everything the LLM writes is untrusted input. The session's access is what keeps that safe.
