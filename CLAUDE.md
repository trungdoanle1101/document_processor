# Working with Claude on this repository

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

- The regulation documents are restricted. Never commit documents or question sets;
  keep them under the git-ignored `data/`. Never send them to cloud services without
  explicit approval.
- Never commit `.env`. Never log tokens; store only their hashes.
- Log ids, versions and codes only, never document text.
- Keep code database-agnostic: no driver-specific internals.