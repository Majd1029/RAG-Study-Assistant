"""FastAPI interface to the same pipeline, for programmatic use.

NOT what the hosted demo runs — Streamlit owns the port there, and this keeps
its index in a single module-level variable shared by every caller. That is
fine for a local single-user service and wrong for a public one; if you expose
this, key the store by session or user first.

Run locally:

    uvicorn api:app --reload
    # docs at http://127.0.0.1:8000/docs
"""
import io
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ingest import ingest_pdfs
from rag_pipeline import build_prompt, load_llm, retrieve

app = FastAPI(title="RAG Study Assistant API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_store = None
_history: list[dict] = []


class _Upload:
    """Adapter so ingest_pdfs can take FastAPI uploads as well as Streamlit ones."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return io.BytesIO(self._data).getbuffer()


@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global _store, _history
    try:
        wrapped = [_Upload(f.filename, await f.read()) for f in files]
        _store, n_chunks, skipped = ingest_pdfs(wrapped)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _history = []
    return {"message": "PDFs processed", "num_chunks": n_chunks, "skipped": skipped}


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if _store is None:
        raise HTTPException(status_code=409, detail="Upload PDFs first.")

    docs = retrieve(_store, question)
    if not docs:
        return {"question": question, "answer": "Not found in the notes.",
                "sources": []}

    tok, model = load_llm()
    inputs = tok(build_prompt(tok, question, docs, _history),
                 return_tensors="pt", truncation=True, max_length=2048)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    answer = tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    _history.append({"question": question, "answer": answer})
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"pdf": d.metadata.get("source_pdf"), "page": d.metadata.get("page")}
            for d in docs
        ],
    }


@app.post("/reset/")
async def reset():
    global _history
    _history = []
    return {"message": "Conversation memory cleared."}
