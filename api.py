from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import shutil
import os
from ingest import ingest_pdfs
from rag_pipeline import get_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Study Assistant API")

# Allow CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    file_paths = []
    for f in files:
        temp_path = f"temp_{f.filename}"
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        file_paths.append(temp_path)

    vectordb = ingest_pdfs(file_paths)
    return {"message": "PDFs processed successfully", "num_chunks": len(vectordb.index)}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    answer = get_answer(question)
    return {"question": question, "answer": answer}
