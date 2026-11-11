# Retrieval as a graph of stages

Status: design discussion, nothing built. Branch: `features/retrieval-redesign`.

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
| 2 | **Current:** nothing deleted, no replaced version | guarantee |
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

Every stage that adds candidates (source, expand) receives the filter.

## 5. Statements and the executor (the SQLAlchemy pattern)

**A statement describes a search; an executor runs it**, like `select()` and
`session.execute()`.

```text
stmt = search(question).retrieve(dense, sparse).merge(rrf).rerank(cross_encoder).take(10)
passages = searcher_for(asker).execute(stmt)
```

- **The asker is bound to the executor, not the statement.** Why: a statement can't forget the
  access filter. The executor adds it to every source and puts the currency gate at the end.
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

## 6. Validation

| When | What | Why |
|---|---|---|
| Before any work ("compile") | stage order, the index supports the filter, the tools exist, limits fit (k ≤ depth) | nothing is spent on a statement that would fail |
| During the run | empty results, wrong score counts, a tool failing | only known as it runs |

mypy covers statements written in code. Statements built at run time (from config, or by an
agent) still need the compile check, so the executor always runs it.

## 7. Loops

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

## 8. Agents that recommend decisions

Retrieval is the agent's evidence layer. The guarantees carry up to the recommendation unchanged.

```text
decision agent (controller)
  1. frame     the decision → what must be known (options, constraints, criteria)
  2. gather    one statement per need  ──→ executor bound to the user
  3. weigh     options against evidence (the LLM's work)
  4. check     is every claim backed by a passage? does any evidence conflict?
  5. answer    recommendation + cited evidence, or "not enough evidence"
```

| Concern | Provided by |
|---|---|
| No recommendation rests on, or hints at, hidden knowledge | the executor bound to the user |
| No deciding on outdated knowledge | the currency gate on every statement |
| "Why did it recommend this?" | passages with origin; statements logged and replayable |
| Coverage: every constraint that applies, counter-evidence | different statements per need ("supports A", "argues against A", "everything in scope") |
| Knowing when not to recommend | scores and gates, so the agent can abstain |
| Is it improving? | fixed statements per step, evaluated one step at a time |

The framework doesn't make step 3 correct. It guarantees the reasoning starts from permitted,
current, traceable evidence. Step 4 checks the output against that evidence, and that check is
where most of the remaining safety comes from.

## 9. Scope of the first version

Build only the stage kinds that basic search needs: **retrieve, merge, rerank, take**, with the
filter expression and the executor's frame. Add rewrite, expand, prune and the controller when
something uses them. Why: the type-state structure is the costly part to get right; adding
stage kinds afterwards is cheap.

## 10. Open questions

- **Filter expression:** AND / OR / NOT across attributes and document ids, compiled per backend
  (Qdrant must/should/must_not; an in-memory predicate), with unsupported checks at compile
  time. Methods rather than operator overloading, for strict mypy.
  - Alternative to weigh: one `access_keys` attribute per chunk (`group:<id>`,
    `collection:<id>`) turns the access OR into one any-of, at the cost of a denormalised
    attribute to keep in sync.
- **Chunk schema:** one declaration of attribute names, types and indexes, used by chunks and
  by backends (Qdrant payload indexes follow from it).
- **Typed ids:** `uuid.UUID` for documents instead of `str`.
- **Currency during re-index:** while a new version waits to be indexed, hide the document, or
  show the last indexed version?
- **Scores:** one `score` per passage (rerank if present, else index), comparable only within a
  response? What do candidates carry between stages?
- **Reranker latency:** depth 50 on CPU may take seconds; the latency budget sets the default.
- **The agent step API:** how a statement starts from earlier candidates, and what the
  controller sees.
- **Migration:** build alongside the old API, port callers (server indexing handlers, search,
  evaluation, tests), then delete the old one.
