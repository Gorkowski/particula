# Chat

Chat models are **conversational generalists**.  
In Particula they are used for:

- interactive tutorials & “explain‑this‑output” prompts,
- translating user intent to build small function or class calls.

Each chat request is wrapped in a **vector‑RAG layer** that pulls
the most relevant Particula API & docs snippets before the prompt
reaches the LLM.

```mermaid
graph TB
    U["User"] -->|prompt| REAS["Chat Model"]
    REAS -->|tool call| VS[(Vector Store)]
    VS --> |API/Examples/Theory| REAS
    REAS -->|final answer| U
```


## GPT‑4o

GPT‑4o (o for **omni**) is a multimodal model that natively handles text,
images, and (soon) audio & video inside **one** architecture.

- 128 k‑token context
- GPT‑4‑Turbo parity on code & English 
- better vision & non‑English.

**Use for:** integrated visual or mixed‑media reasoning (diagrams, photos, future A/V).

## GPT‑4.1

April 2025 upgrade to 4o.

- 1 M‑token context window
- +21 % coding accuracy vs 4o 
- 26 % cheaper ops.

**Use for:** very‑long‑context refactors, legal/scientific deep‑dives, multi‑step agents.

## GPT‑4.1mini & GPT‑4.1nano

- **mini** – ½ latency, 83 % cheaper than 4o, still beats it on many tasks.  
- **nano** – smallest & fastest of 4.1.

**Use for:** mini → balanced power/cost; nano → ultra‑light, real‑time or mobile agents.

---

Reference:

- [OpenAI Docs](https://platform.openai.com/docs/models)
- [OpenAI 4.1 press release](https://openai.com/index/gpt-4-1/)
