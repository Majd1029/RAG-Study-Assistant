from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
import json
import os

MEMORY_FILE = "chat_memory.json"

# Robust chat memory loader
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)

# Main RAG Q&A function
def get_answer(question, index_path="faiss_index", k=4):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    docs = retriever.vectorstore.similarity_search(question, k=k)

    if not docs:
        return "I couldn't find this in your notes."

    context = "\n\n".join(
        f"(PDF: {d.metadata.get('source_pdf', 'N/A')} | Page {d.metadata.get('page', 'N/A')}) {d.page_content}"
        for d in docs
    )

    memory = load_memory()
    chat_history = "\n".join([f"Q: {m['question']}\nA: {m['answer']}" for m in memory])

    prompt = f"""
You are a study assistant. Answer ONLY using the context below.
If the answer is not in the context, say "Not found in the notes."

Previous conversation:
{chat_history}

Current Context:
{context}

Question: {question}

Answer with citations like (PDF: ..., Page X).
"""
    llm = ChatOllama(model="mistral", temperature=0)
    response = llm.invoke(prompt)
    answer = response.content

    memory.append({"question": question, "answer": answer})
    save_memory(memory)

    return answer
