"""Retrieval + generation with a small open model loaded in-process.

Replaces the original `ChatOllama(model="mistral")` call: Ollama is a local
daemon and does not exist on a hosted runner, so the original pipeline cannot
run there at all.

Why Qwen2.5-0.5B-Instruct at bfloat16. Measured on CPU with a 199-token
retrieved context and a forced 80-token answer:

    config                    peak RSS   prefill   decode       total
    Qwen 0.5B   float32        2686 MB      --         --         --
    Qwen 0.5B   bfloat16       1568 MB      6.4s   18.0 tok/s    10.8s
    SmolLM2-360M float32       2027 MB      1.1s    7.4 tok/s    11.9s

float32 does not fit Streamlit Community Cloud's ~2.7GB ceiling once
Streamlit's own ~200MB is added. bfloat16 does, with ~1GB to spare, and is
*faster* at decode despite the CPU having no native bf16 matmul: token
generation is memory-bandwidth-bound, so halving bytes-per-weight wins more
than the upcast costs. It loses only on prefill, which is compute-bound.
"""
import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
MAX_CONTEXT_TOKENS = 2048

_lock = threading.Lock()
_tok = None
_model = None

SYSTEM = (
    "You are a study assistant. Answer ONLY from the context provided. "
    'If the answer is not in the context, reply exactly: "Not found in the notes." '
    "Cite every claim as (PDF: <filename>, Page <n>). Be concise."
)


def load_llm():
    """Loaded once per process. transformers 5.x renamed `torch_dtype` to
    `dtype`; the old name still works but warns and is scheduled for removal."""
    global _tok, _model
    with _lock:
        if _model is None:
            _tok = AutoTokenizer.from_pretrained(LLM_MODEL)
            _model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            _model.eval()
    return _tok, _model


def retrieve(vectordb, question, k=4):
    return vectordb.similarity_search(question, k=k)


def build_prompt(tok, question, docs, history=None):
    context = "\n\n".join(
        f"(PDF: {d.metadata.get('source_pdf', 'N/A')} | "
        f"Page {d.metadata.get('page', 'N/A')}) {d.page_content}"
        for d in docs
    )
    convo = "".join(
        f"Q: {t['question']}\nA: {t['answer']}\n" for t in (history or [])[-2:]
    )
    user = (f"Previous conversation:\n{convo}\n" if convo else "") + (
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def stream_answer(vectordb, question, history=None, k=4):
    """Yield tokens as they are generated, so the page shows progress instead of
    sitting blank for ~11s. Returns (iterator, retrieved_docs)."""
    docs = retrieve(vectordb, question, k=k)
    if not docs:
        def _empty():
            yield "I couldn't find this in your notes."
        return _empty(), []

    tok, model = load_llm()
    prompt = build_prompt(tok, question, docs, history)
    inputs = tok(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS
    )

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            streamer=streamer,
        ),
        daemon=True,
    ).start()
    return streamer, docs
