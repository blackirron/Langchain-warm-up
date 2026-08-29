# LangChain Warm-Up

A small collection of experiments I built while getting familiar with **LangChain and local AI agents**.

The goal was simple: understand how the pieces fit together instead of jumping straight into a big agent framework.

### What’s inside

* **Document loading & chunking** — loading web content and splitting it into smaller chunks.
* **Vector search with FAISS** — creating a local embedding index using `nomic-embed-text`.
* **RAG** — retrieving relevant documents and using a local LLM to answer from that context.
* **Conversational RAG** — adding chat history so follow-up questions make sense.
* **ReAct agent** — giving the model a retrieval tool and letting it decide when to use it.

Everything runs locally through **Ollama**, using `qwen2.5-coder:1.5b` for generation and `nomic-embed-text` for embeddings.

### Run

Make sure Ollama is installed and the required models are available:

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull nomic-embed-text
```

Install the Python dependencies, then run the scripts individually:

```bash
python document_loader.py
python vector_store.py
python rag_chain.py
python rag_chain_main.py
python agent_chain.py
```

### Why this repo?

This is mostly a **learning project**. A record of me figuring out how local LLMs, retrieval, memory, and agents actually work together.

Nothing here is meant to be production-ready. The point was to build each part, break it, fix it, and understand what was happening under the hood.
