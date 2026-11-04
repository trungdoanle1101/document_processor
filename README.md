# document_processor

A framework that turns structured documents (.docx, plain text) into chunks you can retrieve. It covers parsing, outline profiles and chunking, then indexing (in memory or Qdrant), search with fusion and reranking, evaluation, and grounded answers with citations.

- `src/document_processor/` is the framework.
- `src/knowledge_base/` is a tentative app built on it: a CLI and an OpenAI-compatible API for Open WebUI, with access groups and versioned documents. See [its README](src/knowledge_base/README.md).
