# 🧠 GenAI Mindmap

An interactive knowledge map of the Generative AI landscape — covering prompting, RAG, agents, fine-tuning, infrastructure, evaluation, safety, and applications.

**Live site:** [genai4a11.github.io](https://genai4a11.github.io)

---

## What's inside

- **337 nodes** across 5 clusters: Foundations, Building, Production, Governance, and Applications
- **Rich content panels** on every major node — use cases, diagrams, Python code examples, and pro tips
- **Key questions by role** — PM and Engineer questions on all 64 topic nodes; Leader + PM + Engineer on root and cluster nodes
- **30 concept deep-dive pages** — full articles with comparison tables, method cards, code examples, and references
- **Further reading links** on 20 high-value nodes — canonical papers, primary docs, and key blog posts
- **Real-time search** across all nodes
- **Resizable side panel** with width persistence

---

## How to use

- **Click** any node to open its detail panel
- **Click the ▸ chip** on a node to expand its subtopics
- **Search** using the bar at the top
- **Click `📄 Deep dive →`** in the panel to open the full concept article for that topic
- **Drag the panel edge** left or right to resize it
- **Use the toolbar** (bottom-left) to navigate between Clusters, Domains, and full map views

---

## Concept deep-dive pages

Standalone articles covering the most complex topics in depth. Each page includes comparison tables, method cards, code examples, callouts, tools grids, and references.

### Inference & Architecture
| Page | Topics covered |
|------|----------------|
| [KV Cache Mechanics](concepts/kv-cache.html) | Cache memory math, GQA/MLA, eviction strategies, PagedAttention, prefix caching |
| [Quantization](concepts/quantization.html) | GPTQ, AWQ, SmoothQuant, GGUF, PTQ vs QAT, hardware support |
| [Transformer Architecture](concepts/transformer-arch.html) | MHA, GQA, FFN/SwiGLU, RoPE, pre-norm, architecture variants |
| [LLM Internals](concepts/llm-internals.html) | Tokenization, context window, sampling strategies, logprobs, chat templates |
| [Advanced Reasoning](concepts/advanced-reasoning.html) | CoT, PRM, tree search, GRPO, test-time compute scaling |
| [LLM Serving](concepts/serving.html) | TTFT/TBT, continuous batching, parallelism, speculative decoding |

### Training & Fine-Tuning
| Page | Topics covered |
|------|----------------|
| [PEFT Methods](concepts/peft-methods.html) | LoRA, QLoRA, prefix tuning, adapters, rank selection, memory table |
| [LLM Alignment](concepts/alignment.html) | RLHF, DPO, Constitutional AI, reward models, training pipeline |
| [Training Techniques](concepts/training-tech.html) | Mixed precision, FSDP, DeepSpeed ZeRO, tensor/pipeline parallelism |
| [Synthetic Data](concepts/synthetic-data.html) | Self-instruct, distillation, rejection sampling, Magpie, model collapse |

### Retrieval & RAG
| Page | Topics covered |
|------|----------------|
| [Text Embeddings](concepts/embeddings.html) | Models, MTEB, Matryoshka, fine-tuning, cross-encoder reranking |
| [Retrieval Technology](concepts/retrieval-tech.html) | BM25, FAISS, HNSW, hybrid RRF, vector DB comparison |
| [Advanced RAG](concepts/advanced-rag.html) | HyDE, chunking, reranking, hybrid search, multi-hop, RAGAS |
| [Vector Databases](concepts/vector-dbs.html) | pgvector, Qdrant, Weaviate, Pinecone, Chroma — selection guide |
| [Golden Datasets](concepts/golden-datasets.html) | Building eval datasets, quality criteria, annotation pipelines |

### Agents & Orchestration
| Page | Topics covered |
|------|----------------|
| [Agent Frameworks](concepts/agent-frameworks.html) | LangGraph, CrewAI, AutoGen, Pydantic AI — comparison and decision guide |
| [Agent Planning](concepts/agent-planning.html) | ReAct, plan-and-execute, Reflexion, Tree of Thoughts, memory |
| [Agent Memory Systems](concepts/agent-memory.html) | Working/episodic/semantic/procedural memory, MemGPT, production patterns |
| [Multi-Agent Systems](concepts/multi-agent.html) | Orchestrator-subagent, parallel, debate, trust, observability |
| [Tool Use & Function Calling](concepts/tool-use.html) | Function calling loop, schema design, parallel calls, strict mode |

### Models & Prompting
| Page | Topics covered |
|------|----------------|
| [Frontier LLM Models](concepts/frontier-models.html) | GPT-4o, Claude, Gemini, Llama 3, selection guide, multi-provider routing |
| [Prompt Engineering Fundamentals](concepts/basic-prompting.html) | System prompts, few-shot, CoT, role prompting, format control |
| [Programmatic Prompting](concepts/programmatic-prompting.html) | DSPy, LCEL, MIPRO, OPRO, APE, template engines |
| [Output Control](concepts/output-control.html) | JSON mode, instructor, Outlines constrained decoding, grammar |
| [Vision-Language Models](concepts/vision-language.html) | CLIP, LLaVA, GPT-4V, tiled tokenization, multimodal pipelines |

### Safety, Evaluation & Governance
| Page | Topics covered |
|------|----------------|
| [LLM Safety Techniques](concepts/safety-tech.html) | Constitutional AI, red-teaming, guardrails, jailbreak defenses |
| [LLM Evaluation in Practice](concepts/evals-practice.html) | Eval funnel, LLM-as-judge, golden sets, framework comparison |
| [LLM Benchmarks Explained](concepts/benchmarks.html) | MMLU, SWE-Bench, Chatbot Arena, saturation, selection guide |
| [LLM Reliability Engineering](concepts/reliability.html) | Hallucination mitigation, grounding, uncertainty, fallback patterns |

### Production & Infrastructure
| Page | Topics covered |
|------|----------------|
| [MLOps for LLMs](concepts/mlops.html) | Prompt CI/CD, model registry, drift detection, canary deploys, cost |
| [AI Hardware Guide](concepts/hardware.html) | H100 vs A100 vs MI300X, cloud vs on-prem, memory math, interconnects |

---

## Acknowledgements

Thanks to the following people for their early feedback and suggestions:

- **Cemalettin Ozturk**
- **Antonio Penta**
- **Shamsul Hassan**
- **Adam Napora**

---

## License

This project is licensed under the [MIT License](LICENSE) — free to use, share, and modify with attribution.

---

## Clap / Like Widget

Each page includes a **LikeBtn** clap widget so visitors can indicate what they find useful. Aggregated counts per page are stored on [LikeBtn.com](https://likebtn.com).

The widget is loaded via `assets/js/clap.js`, which:
- Injects a `<span class="likebtn-wrapper">` into any `[data-clap]` placeholder element.
- Uses `window.location.pathname` as a stable per-page identifier.
- Loads the LikeBtn script only once per page.

To **disable** the widget on a specific page, remove the `<div data-clap ...></div>` element and the `<script src="/assets/js/clap.js"></script>` tag from that page.

To **remove** it from the entire site, delete `assets/js/clap.js` and remove all `data-clap` placeholders and the script tags referencing `clap.js`.

---

## Built by

[Deepak Mehta](https://www.linkedin.com/in/deepakmehta79) — built as a personal learning resource, open to the community.

Feedback welcome via [GitHub Issues](https://github.com/genai4a11/genai4a11.github.io/issues/new?title=Feedback&body=**Type**%3A%20%5B%20%5D%20Missing%20node%20%20%5B%20%5D%20Wrong%20content%20%20%5B%20%5D%20Suggestion%0A%0A**Details**%3A).
