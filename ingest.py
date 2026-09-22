"""PDF ingestion -> chunks -> FAISS.

The index is returned, never written to disk. The original wrote `faiss_index/`
and `chat_memory.json` into the working directory, which is fine on a laptop and
wrong on a shared host: one process serves every visitor, so visitor B would
have been answering questions against visitor A's documents. Everything here
lives in Streamlit session state instead and disappears with the session.

Embeddings use all-MiniLM-L6-v2 (~90MB) rather than the original
all-mpnet-base-v2 (~420MB): roughly 5x faster on CPU, for a small retrieval
quality cost that is not visible at this corpus size.
"""
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_PAGES = int(os.getenv("MAX_PAGES", "120"))

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def ingest_pdfs(uploaded_files, progress=None):
    """uploaded_files: Streamlit UploadedFile objects. Returns a FAISS store.

    Raises ValueError with a readable message when a PDF yields no text, which
    is what happens with scanned documents that need OCR first.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    all_chunks, total_pages, skipped = [], 0, []

    for n, uf in enumerate(uploaded_files):
        if progress:
            progress(n / len(uploaded_files), f"Reading {uf.name}...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
        except Exception as e:
            skipped.append(f"{uf.name} ({e})")
            continue
        finally:
            os.unlink(tmp_path)

        if not any(p.page_content.strip() for p in pages):
            skipped.append(f"{uf.name} (no extractable text — scanned?)")
            continue

        total_pages += len(pages)
        if total_pages > MAX_PAGES:
            raise ValueError(
                f"That is over {MAX_PAGES} pages. This demo runs on a free CPU "
                "tier; please try a smaller selection."
            )

        chunks = splitter.split_documents(pages)
        for c in chunks:
            c.metadata["source_pdf"] = uf.name
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "No extractable text found. "
            + ("Skipped: " + "; ".join(skipped) if skipped else "")
        )

    if progress:
        progress(0.9, f"Embedding {len(all_chunks)} chunks...")
    store = FAISS.from_documents(all_chunks, get_embeddings())
    return store, len(all_chunks), skipped
