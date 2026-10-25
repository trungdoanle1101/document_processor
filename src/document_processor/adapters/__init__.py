from document_processor.adapters.plain_text import PlainTextAdapter
from document_processor.adapters.docx.adapter import DocxAdapter
from document_processor.adapters.adapter import Adapter
from document_processor.core.registry import Registry


__all__ = ["ADAPTERS", "Adapter", "DocxAdapter", "PlainTextAdapter"]


ADAPTERS: Registry[Adapter] = Registry("adapter", {
    ".docx": DocxAdapter(),
    ".txt": PlainTextAdapter()
})
