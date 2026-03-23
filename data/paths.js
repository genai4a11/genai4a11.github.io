const PATHS={
  foundations:[
    {id:'math_foundations',note:'Linear algebra, calculus, probability — the math that underpins everything'},
    {id:'python_ecosystem',note:'NumPy, Pandas, Matplotlib — your daily scientific computing tools'},
    {id:'pytorch_basics',note:'Tensors, autograd, and the training loop — essential before any LLM work'}
  ],
  ml_core_domain:[
    {id:'neural_nets',note:'Backprop, activations, LayerNorm — how models learn representations'},
    {id:'optimization',note:'Adam, LR scheduling — making your training converge reliably'},
    {id:'training_tech',note:'Mixed precision, gradient checkpointing, DeepSpeed for scale'},
    {id:'regularization',note:'Dropout, weight decay — preventing overfitting'}
  ],
  transformers_domain:[
    {id:'attention',note:'Self-attention is the core innovation — master Flash Attention & Paged Attention'},
    {id:'transformer_arch',note:'Encoder-only vs decoder-only vs enc-dec — which to use when'},
    {id:'pos_encoding',note:'RoPE and ALiBi — how models know token order and handle long contexts'},
    {id:'kv_cache',note:'GQA, speculative decoding — efficiency tricks for inference'}
  ],
  llms:[
    {id:'llm_internals',note:'Tokenization, context window, scaling laws — understand before you use'},
    {id:'open_models',note:'Start with Llama 3 or Mistral locally — free, fine-tunable'},
    {id:'frontier_models',note:'GPT-4o, Claude 3.5 — when API quality justifies the cost'}
  ],
  prompting:[
    {id:'basic_prompting',note:'Zero-shot, few-shot, system prompts — the foundation'},
    {id:'advanced_reasoning',note:'Chain-of-thought, extended thinking, self-consistency — unlock reasoning'},
    {id:'output_control',note:'JSON mode, Instructor, Outlines — get structured typed data back'},
    {id:'programmatic_prompting',note:'DSPy — optimize prompts automatically with labeled data'}
  ],
  rag:[
    {id:'embeddings_topic',note:'How semantic similarity works — BGE or OpenAI embeddings to start'},
    {id:'retrieval_tech',note:'Dense + BM25 hybrid + HyDE — find the most relevant chunks'},
    {id:'vector_dbs',note:'Chroma locally → Qdrant or Pinecone in production'},
    {id:'post_retrieval',note:'Cohere Rerank + context compression — sharpen precision'},
    {id:'advanced_rag',note:'LlamaIndex, GraphRAG, agentic RAG — production patterns'}
  ],
  agents:[
    {id:'agent_planning',note:'ReAct, Plan & Execute, reflection — how agents decompose goals'},
    {id:'tool_use',note:'Function calling and MCP — agents interacting with the real world'},
    {id:'agent_memory',note:'Conversation buffer, long-term vector store, entity memory'},
    {id:'agent_frameworks',note:'LangGraph for stateful workflows; CrewAI for role-based teams'},
    {id:'multi_agent',note:'Orchestrator-worker and parallel execution — coordinate many agents'}
  ],
  finetuning:[
    {id:'data_prep',note:'Data quality beats model quality — curate carefully'},
    {id:'peft_methods',note:'LoRA and QLoRA — adapt models by training <1% of weights'},
    {id:'ft_tools',note:'Unsloth + TRL for fast training; Axolotl for YAML-configured runs'},
    {id:'alignment',note:'DPO and ORPO to steer model behavior toward preferences'}
  ],
  multimodal:[
    {id:'vision_language',note:'GPT-4V, LLaVA, Qwen-VL — understand images with LLMs'},
    {id:'audio_models',note:'Whisper for ASR, ElevenLabs / Kokoro for TTS pipelines'},
    {id:'image_gen',note:'Stable Diffusion, Flux — open-source text-to-image'},
    {id:'video_models',note:'Frontier research — Sora and open alternatives like Wan2.1'}
  ],
  infra:[
    {id:'serving',note:'vLLM for cloud, Ollama for local — deploy with one command'},
    {id:'quantization',note:'GGUF, AWQ, BitsAndBytes — shrink models to fit your hardware'},
    {id:'cloud_deploy',note:'HF Spaces, Modal Labs, Replicate — go serverless in minutes'},
    {id:'hardware',note:'NVIDIA H100/A100, Apple MLX, AMD ROCm — know your silicon'}
  ],
  eval:[
    {id:'rag_eval',note:'RAGAS and TruLens — measure faithfulness and relevance first'},
    {id:'monitoring',note:'Langfuse or W&B Weave — trace every LLM call in production'},
    {id:'benchmarks',note:'MMLU, HumanEval, MT-Bench — compare models on standard tasks'},
    {id:'evals_practice',note:'Evals-first workflow — design your test suite before writing code'}
  ],
  safety:[
    {id:'safety_tech',note:'Safety Techniques — red teaming, guardrails, and constitutional AI'},
    {id:'prompt_injection',note:'Understand the attack surface before building defences'},
    {id:'red_teaming',note:'Proactively probe for jailbreaks and unsafe behaviours'},
    {id:'guardrails',note:'Llama Guard, NeMo Guardrails — input/output filters in production'}
  ],
  meta_applications:[
    {id:'rag_systems',note:'Document Q&A — the most common enterprise GenAI use case'},
    {id:'code_assist',note:'Cursor, Copilot, Aider — AI in your editor today'},
    {id:'structured_output_app',note:'Instructor and Outlines — extract typed data reliably'},
    {id:'database_query',note:'Text-to-SQL for self-serve analytics and data access'},
    {id:'voice_agents',note:'LiveKit + Whisper + TTS for real-time < 300ms voice pipelines'},
    {id:'doc_processing',note:'Unstructured.io, Docling — parse PDFs and tables for RAG'}
  ],
  framework_tools:[
    {id:'mlops',note:'MLflow, DVC, HuggingFace Hub — track experiments and versions'},
    {id:'integration_std',note:'LiteLLM and OpenAI Compat API — swap LLM providers freely'},
    {id:'dev_frameworks',note:'Streamlit, Gradio, FastAPI — ship demos and APIs fast'}
  ],
  // Topic-level paths (g:2 nodes)
  agent_planning:[
    {id:'react_agent',note:'The foundation — Reason, Act, Observe loop for all tool-using agents'},
    {id:'plan_execute',note:'Separate upfront planning from execution for complex multi-step tasks'},
    {id:'reflection',note:'Add self-critique: agent reviews its own output before finishing'}
  ],
  agent_memory:[
    {id:'st_memory',note:'Start here — sliding window conversation buffer (simplest)'},
    {id:'entity_memory',note:'Track people, orgs, and facts extracted from conversation'},
    {id:'lt_memory',note:'Persist to a vector store so the agent remembers across sessions'}
  ],
  tool_use:[
    {id:'fc_api',note:'JSON schema defines your tools — model decides when and how to call them'},
    {id:'mcp',note:'Anthropic open standard — write once, works in Claude, Cursor, and more'},
    {id:'tool_selection',note:'RAG-over-tools pattern for agents with many available functions'},
    {id:'code_mode',note:'LLM writes code against a typed SDK; Dynamic Worker Loader runs it safely — context-efficient alternative to per-call JSON tool dispatch'}
  ],
  agent_frameworks:[
    {id:'langchain',note:'Start here for prototyping — huge ecosystem, LCEL pipe syntax'},
    {id:'langgraph',note:'Graduate to this for production — stateful graphs with loops'},
    {id:'crewai',note:'Role-based multi-agent teams with task delegation'},
    {id:'autogen',note:'Microsoft conversable agents — great for code execution workflows'},
    {id:'smolagents',note:'HuggingFace minimal framework — best for simple code-first agents'}
  ],
  multi_agent:[
    {id:'sequential_chain',note:'Start here — linear pipelines are the simplest agentic workflow'},
    {id:'orchestrator',note:'Central planner routes subtasks to the right specialist agent'},
    {id:'parallel_agents',note:'Fan-out pattern — run independent subtasks concurrently'},
    {id:'event_driven_agent',note:'Trigger agents from queues, webhooks, or schedules'},
    {id:'hitl',note:'Pause for human approval on high-risk or irreversible actions'}
  ],
  attention:[
    {id:'self_attention',note:'Core mechanism — Q·Kᵀ/√d scoring between all token pairs'},
    {id:'multihead_attn',note:'Run H attention heads in parallel, each capturing different patterns'},
    {id:'flash_attn',note:'Drop-in replacement — same result, 2-4× faster, much less memory'},
    {id:'cross_attn',note:'Decoder attending to encoder — used in T5 and multimodal models'}
  ],
  kv_cache:[
    {id:'paged_attn',note:'Non-contiguous KV blocks — the core innovation behind vLLM'},
    {id:'gqa',note:'Grouped-query attention — used in Llama 3 and Mistral to cut KV size'},
    {id:'spec_dec',note:'Draft + verify — get 2-3× faster inference with no quality loss'}
  ],
  peft_methods:[
    {id:'lora',note:'Start here — train <0.2% of weights with rank decomposition matrices'},
    {id:'qlora4bit',note:'4-bit quantized LoRA — fine-tune 70B models on a single GPU'},
    {id:'prefix_tuning',note:'Prepend soft tokens to input — more parameter-efficient than LoRA'},
    {id:'ia3',note:'Scale activations with learned vectors — extreme efficiency for few-shot'}
  ],
  ft_tools:[
    {id:'trl',note:'SFTTrainer + DPOTrainer — the standard HF training library for fine-tuning and alignment'},
    {id:'unsloth',note:'2-5x faster LoRA training via hand-written CUDA kernels — same results, less VRAM'},
    {id:'axolotl',note:'YAML-configured fine-tuning wrapper — LoRA/QLoRA on any HF model with one config file'},
    {id:'llama_factory',note:'Web UI + CLI for fine-tuning 100+ models with built-in datasets'}
  ],
  alignment:[
    {id:'rlhf',note:'The original alignment method — SFT + reward model + PPO'},
    {id:'dpo',note:'Simpler alternative to RLHF — no reward model, one training phase'},
    {id:'orpo',note:'Single-stage SFT + alignment — newest and simplest approach'},
    {id:'rlaif',note:'Replace human labellers with an LLM — scale feedback generation'}
  ],
  embeddings_topic:[
    {id:'st_lib',note:'Start here — open-source, runs locally, 200+ pretrained models'},
    {id:'bge_rag',note:'BGE-large or E5-large — top of MTEB leaderboard, free'},
    {id:'oai_emb',note:'text-embedding-3-small — best price/quality for production'},
    {id:'cohere_emb',note:'Cohere Embed v3 — best for multilingual and int8 efficiency'}
  ],
  retrieval_tech:[
    {id:'dense_retrieval',note:'Embed query + docs, find nearest vectors — semantic understanding'},
    {id:'bm25',note:'Keyword matching — essential for product names, codes, exact terms'},
    {id:'hybrid_search',note:'Dense + BM25 fused via RRF — best recall in production'},
    {id:'hyde',note:'Generate a hypothetical answer first, then embed it — boosts recall'},
    {id:'colbert',note:'Late interaction — token-level MaxSim for better accuracy than bi-encoders'},
    {id:'multi_query',note:'Rewrite query N ways, union results — broader coverage'}
  ],
  programmatic_prompting:[
    {id:'prompt_versioning',note:'Version prompts like code — A/B test, track metrics, rollback'},
    {id:'prompt_caching',note:'Cache shared prefixes — 90% cheaper, 2× faster on repeated calls'},
    {id:'dspy',note:'Optimize prompts automatically with labeled examples'}
  ],
  serving:[
    {id:'vllm',note:'Start here for cloud serving — paged attention + continuous batching'},
    {id:'llm_router',note:'Route to cheap model for simple queries — 70% cost reduction'},
    {id:'semantic_cache',note:'Cache semantically similar queries — massive savings for high traffic'},
    {id:'batch_api',note:'50% off for async workloads — evals, labelling, bulk generation'},
    {id:'streaming',note:'Stream tokens for responsive UX — essential for chat interfaces'}
  ],
  safety_tech:[
    {id:'prompt_injection',note:'Understand the attack — then build privilege separation defences'},
    {id:'red_teaming',note:'Actively probe for jailbreaks before shipping'},
    {id:'guardrails',note:'Llama Guard, NeMo Guardrails — input/output filters'},
    {id:'const_ai',note:'Constitutional AI — principles-guided self-critique'}
  ],
  multi_agent:[
    {id:'sequential_chain',note:'Start here — linear pipelines are the simplest agentic workflow'},
    {id:'orchestrator',note:'Central planner routes subtasks to specialist agents'},
    {id:'parallel_agents',note:'Fan-out — run independent subtasks concurrently'},
    {id:'event_driven_agent',note:'Trigger agents from queues, webhooks, or schedules'},
    {id:'hitl',note:'Pause for human approval on high-risk or irreversible actions'}
  ],
  doc_processing:[
    {id:'chunking',note:'Start here — chunking strategy is the biggest RAG quality lever'},
    {id:'contextual_retrieval',note:'Add context summary to each chunk before embedding — 67% better recall'},
    {id:'unstructured',note:'Parse PDFs, HTML, DOCX with Unstructured.io'},
    {id:'docling',note:'IBM Docling for advanced table extraction and OCR'}
  ],
  chunking:[
    {id:'fixed_size',note:'Simplest baseline — start here, no dependencies'},
    {id:'sentence_window',note:'Preserves sentence boundaries — good for Q&A'},
    {id:'semantic_chunking',note:'Topic-aware splits — best for mixed content'},
    {id:'parent_child',note:'Precision + context — the production standard'},
    {id:'late_chunking',note:'Full-doc context in every chunk — for single-topic docs'},
    {id:'proposition_chunking',note:'Atomic facts — highest precision, highest cost'},
    {id:'agentic_chunking',note:'LLM decides boundaries — best for mixed-format docs'}
  ],
  // System Design domain paths
  sysdesign:[
    {id:'decision_fwk',note:'Start here — most costly mistakes come from choosing the wrong approach'},
    {id:'evals_practice',note:'Evals-first: define success before writing a line of pipeline code'},
    {id:'compound_ai',note:'Think in systems — pipelines of models, caches, and validators'},
    {id:'frontier_layer',note:'Understand what test-time compute and long context change about system design'}
  ],
  decision_fwk:[
    {id:'rag_vs_ft',note:'The most common wrong decision in enterprise AI — use this decision tree'},
    {id:'agent_vs_pipe',note:'Agents are 50-100x more expensive — use pipelines unless you genuinely need dynamic steps'},
    {id:'model_select',note:'Benchmark YOUR task — generic benchmarks are poor predictors of real performance'},
    {id:'build_vs_buy',note:'API first, OSS if privacy/cost forces it, custom only as last resort'}
  ],
  evals_practice:[
    {id:'eval_design',note:'Define the success metric before touching the system'},
    {id:'golden_sets',note:'20 hand-curated examples beats 500 auto-generated ones'},
    {id:'prompt_regression',note:'Run evals on every prompt change — treat prompts like code'},
    {id:'online_eval',note:'Sample live traffic continuously — offline evals alone will miss distribution shift'}
  ],
  compound_ai:[
    {id:'cost_quality_triangle',note:'Make trade-offs explicit — you cannot optimise all three simultaneously'},
    {id:'latency_budget',note:'Allocate milliseconds per component before you build'},
    {id:'fallback_chains',note:'Every LLM call needs a fallback — never a single point of failure'},
    {id:'system_eval',note:'Component metrics lie — measure end-to-end quality'}
  ],
  frontier_layer:[
    {id:'test_time_compute',note:'o1/o3/Claude thinking — reasoning time replaces model size for hard problems'},
    {id:'long_ctx_impact',note:'Long context changes RAG tradeoffs — know when to stuff vs retrieve'},
    {id:'distillation_ft',note:'Frontier model generates data → small model trained on it → 50-100x cheaper'},
    {id:'multimodal_native',note:'Vision + text as first-class inputs changes application design patterns'}
  ],
  data_eng:[
    {id:'data_ingestion',note:'Design metadata schemas and ingestion pipelines before writing any model code'},
    {id:'data_labeling',note:'Build annotation workflows with Label Studio or Argilla for eval and FT data'},
    {id:'synthetic_data',note:'Use GPT-4o to generate and filter training data at scale'},
    {id:'data_governance',note:'Version datasets with DVC, enforce contracts at ingestion boundaries'}
  ],
  data_ingestion:[
    {id:'metadata_design',note:'Schema design is the highest-leverage data decision — do it first'},
    {id:'etl_pipeline',note:'Airflow or Prefect for orchestrating ingestion and chunking'},
    {id:'data_quality',note:'Great Expectations or Pandera for validation gates at ingestion'}
  ],
  data_labeling:[
    {id:'annotation_tools',note:'Label Studio (free) or Scale AI (managed) for annotation workflows'},
    {id:'active_learning',note:'Smart sampling — label the most informative examples first'},
    {id:'label_quality',note:'Measure inter-annotator agreement before trusting your labels'}
  ],
  synthetic_data:[
    {id:'synth_generation',note:'GPT-4o generates instruction-response pairs from seed examples'},
    {id:'synth_quality',note:'Dedup + reward scoring — quality matters more than quantity'},
    {id:'self_instruct',note:'Alpaca/Evol-Instruct: bootstrap thousands of examples from 50 seeds'}
  ],
  data_governance:[
    {id:'data_versioning',note:'DVC versions datasets alongside code — training runs become reproducible'},
    {id:'data_contracts',note:'Pydantic schemas at ingestion boundaries catch breaking changes early'},
    {id:'data_lineage',note:'Track provenance from raw source to training set for compliance'}
  ],
  prod_eng:[
    {id:'execution_models',note:'Understand sync/async/streaming tradeoffs before designing any pipeline'},
    {id:'reliability',note:'Retry + circuit breaker + timeouts — implement all three, not just one'},
    {id:'traffic_cost',note:'Cost-aware routing is usually the highest-ROI production optimisation'},
    {id:'state_sessions',note:'Redis session state + checkpointing make agents resumable and debuggable'},
    {id:'human_oversight',note:'Approval gates before irreversible actions — non-negotiable in production'}
  ],
  reliability:[
    {id:'retry_backoff',note:'Exponential backoff with full jitter — 4 lines of code that prevent outages'},
    {id:'circuit_breaker',note:'Fail fast and recover gracefully — stop hammering a degraded service'},
    {id:'timeout_budget',note:'Allocate time per component — retrieval 2s, LLM 8s, total 12s'}
  ],
  traffic_cost:[
    {id:'cost_routing',note:'Complexity classifier routes cheap queries to cheap models — 5-10× savings'},
    {id:'rate_limiting',note:'Token bucket per tenant prevents one user from exhausting your quota'},
    {id:'budget_guard',note:'Hard spend caps stop runaway agent loops from burning your API budget'}
  ],
  state_sessions:[
    {id:'session_state',note:'Redis-backed sessions with TTL and MAX_TURNS cap'},
    {id:'checkpointing',note:'LangGraph checkpointers let agents resume mid-run after failures'},
    {id:'audit_trail',note:'Immutable call log — essential for compliance and debugging'}
  ],
  human_oversight:[
    {id:'approval_gate',note:'Pause before irreversible actions — send, pay, delete'},
    {id:'override_flow',note:'Define escalation thresholds and SLAs for each decision type'}
  ]
};

