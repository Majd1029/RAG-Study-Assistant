import streamlit as st
from ingest import ingest_pdfs
from rag_pipeline import get_answer
import os

st.set_page_config(page_title="Advanced RAG Study Assistant", layout="wide")
st.title("📚 Advanced RAG Study Assistant")

# --- Multi-PDF upload ---
uploaded_files = st.file_uploader("Upload your study PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for f in uploaded_files:
        path = f"temp_{f.name}"
        with open(path, "wb") as out:
            out.write(f.read())
        file_paths.append(path)

    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            ingest_pdfs(file_paths)
        st.success("PDFs processed!")

# --- Question input ---
question = st.text_input("Ask a question about your notes:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        answer = get_answer(question)
    st.markdown("### 📖 Answer")
    st.write(answer)

# --- Reset memory ---
if st.button("Reset Conversation"):
    if os.path.exists("chat_memory.json"):
        os.remove("chat_memory.json")
    st.success("Conversation memory cleared!")
