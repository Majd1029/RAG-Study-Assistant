"""RAG Study Assistant — Streamlit app for Streamlit Community Cloud.

Nothing a visitor uploads is written to disk or outlives their session: the
FAISS index and the conversation both live in st.session_state.
"""
import streamlit as st

from ingest import ingest_pdfs
from rag_pipeline import LLM_MODEL, stream_answer

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚", layout="wide")

st.title("📚 RAG Study Assistant")
st.caption(
    f"Upload PDFs, ask questions, get answers grounded in them with page "
    f"citations. Retrieval with MiniLM + FAISS, generation by `{LLM_MODEL}` "
    "running on this free CPU tier — no API keys, no quotas."
)

st.session_state.setdefault("vectordb", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("n_chunks", 0)

with st.sidebar:
    st.subheader("1 · Load your notes")
    uploaded = st.file_uploader(
        "Study PDFs", type="pdf", accept_multiple_files=True,
        help="Text-based PDFs. Scanned documents need OCR first.",
    )

    if uploaded and st.button("Process PDFs", type="primary", width="stretch"):
        bar = st.progress(0.0, "Starting...")
        try:
            store, n_chunks, skipped = ingest_pdfs(
                uploaded, progress=lambda f, m: bar.progress(f, m)
            )
            st.session_state.vectordb = store
            st.session_state.n_chunks = n_chunks
            st.session_state.history = []
            bar.empty()
            st.success(f"Indexed {n_chunks} chunks from {len(uploaded)} PDF(s).")
            for s in skipped:
                st.warning(f"Skipped {s}")
        except ValueError as e:
            bar.empty()
            st.error(str(e))
        except Exception as e:
            bar.empty()
            st.error(f"Could not process those PDFs: {e}")

    if st.session_state.vectordb is not None:
        st.caption(f"{st.session_state.n_chunks} chunks indexed for this session.")
        if st.button("Reset conversation", width="stretch"):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.caption(
        "Your PDFs are held in memory for this browser session only — never "
        "written to disk, never shared between visitors, gone when you close "
        "the tab."
    )
    st.markdown("[Source](https://github.com/Majd1029/RAG-Study-Assistant)")

if st.session_state.vectordb is None:
    st.info("Upload one or more text-based PDFs in the sidebar to begin.")
    st.stop()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

question = st.chat_input("Ask a question about your notes")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            stream, docs = stream_answer(
                st.session_state.vectordb, question, st.session_state.history
            )
        # Streamed so text appears within a second or two; a complete answer
        # takes roughly 10-20s on this tier.
        answer = st.write_stream(stream)

        if docs:
            with st.expander(f"Retrieved passages ({len(docs)})"):
                for d in docs:
                    st.markdown(
                        f"**{d.metadata.get('source_pdf')}** — page "
                        f"{d.metadata.get('page')}"
                    )
                    st.caption(d.page_content[:500] + "...")

    st.session_state.history.append({"question": question, "answer": answer})
