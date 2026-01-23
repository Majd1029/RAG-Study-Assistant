from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_pdfs(file_paths, index_path="faiss_index"):
    all_chunks = []

    for path in file_paths:
        loader = PyPDFLoader(path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source_pdf"] = os.path.basename(path)
        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = FAISS.from_documents(all_chunks, embeddings)
    vectordb.save_local(index_path)
    return vectordb
