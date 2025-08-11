# TnT-LLM Implementation Plan

> Implementation of the TnT-LLM paper: "Text Mining at Scale with Large Language Models"  
> Reference: [context.md](file:///home/immortal/Desktop/Taxonomy_generation/context.md)

---

## Project Structure

```
Taxonomy_generation/
├── context.md                    # Paper context & reference
├── plan.md                       # This implementation plan
├── config/
│   ├── settings.yaml             # Global config (API keys, model names, hyperparams)
│   └── use_cases/                # Per-use-case config files
│       ├── intent_detection.yaml
│       └── domain_classification.yaml
├── src/
│   ├── __init__.py
│   ├── config.py                 # Config loader and validation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # Corpus loading (CSV, JSON, JSONL, etc.)
│   │   ├── sampler.py            # Stratified/random sampling for Phase 1 & 2
│   │   └── splitter.py           # Train/val/test split logic
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── summarizer.py         # Stage 1: Document summarization
│   │   ├── taxonomy_generator.py # Stage 2: Generation prompt
│   │   ├── taxonomy_updater.py   # Stage 2: Update prompt chain (SGD loop)
│   │   ├── taxonomy_reviewer.py  # Stage 2: Review prompt
│   │   ├── taxonomy_evaluator.py # Validation-based model selection
│   │   └── pipeline.py           # End-to-end Phase 1 orchestrator
│   ├── phase2/
│   │   ├── __init__.py
│   │   ├── label_assigner.py     # LLM-based pseudo-labeling
│   │   ├── embedder.py           # Document embedding extraction
│   │   ├── classifiers.py        # LogReg, LightGBM, MLP wrappers
│   │   ├── trainer.py            # Training loop with grid search
│   │   └── pipeline.py           # End-to-end Phase 2 orchestrator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── taxonomy_metrics.py   # Coverage, accuracy, relevance
│   │   ├── classification_metrics.py # F1, precision, recall, kappa
│   │   └── llm_evaluator.py      # LLM-as-judge evaluation
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py          # All prompt templates as string constants
│   │   └── builder.py            # Dynamic prompt construction with batching
│   └── utils/
│       ├── __init__.py
│       ├── llm_client.py         # LLM API wrapper (OpenAI/Gemini/local)
│       ├── parsers.py            # XML/JSON taxonomy output parsing
│       ├── io.py                 # File I/O helpers
│       └── logging.py            # Structured logging
├── scripts/
│   ├── run_phase1.py             # CLI entrypoint for Phase 1
│   ├── run_phase2.py             # CLI entrypoint for Phase 2
│   ├── run_full_pipeline.py      # CLI entrypoint for end-to-end
│   └── evaluate.py               # CLI entrypoint for evaluation
├── tests/
│   ├── test_summarizer.py
│   ├── test_taxonomy_generator.py
│   ├── test_taxonomy_updater.py
│   ├── test_label_assigner.py
│   ├── test_classifiers.py
│   └── test_parsers.py
├── notebooks/
│   ├── exploration.ipynb         # Data exploration & visualization
│   └── results_analysis.ipynb    # Results analysis & charts
├── data/
│   ├── raw/                      # Raw input corpus
│   ├── processed/                # Summaries, embeddings
│   ├── taxonomies/               # Generated taxonomy outputs (JSON)
│   └── labels/                   # Pseudo-labels and predictions
├── models/                       # Saved trained classifiers
├── requirements.txt
└── README.md
```

---

## Implementation Phases

### Milestone 0: Project Setup & Infrastructure
**Priority:** 🔴 Critical  
**Estimated effort:** 1 day

| Task | File(s) | Details |
|------|---------|---------|
| Initialize project structure | all dirs | Create directory skeleton |
| Set up conda/venv environment | `requirements.txt` | See dependency list below |
| Create config system | `config/settings.yaml`, `src/config.py` | YAML-based config with env var overrides for API keys |
| Build LLM client abstraction | `src/utils/llm_client.py` | Unified interface supporting OpenAI, Gemini, or local models. Must handle: rate limiting, retries, async batching, temperature control |
| Build output parsers | `src/utils/parsers.py` | Parse XML/JSON taxonomy output from LLM responses. Handle malformed output gracefully with fallbacks |
| Build data loaders | `src/data/loader.py` | Support CSV, JSON, JSONL, plain text directory |

**Dependencies:**
```
openai>=1.0
google-generativeai    # if using Gemini
scikit-learn>=1.3
lightgbm>=4.0
torch>=2.0             # for MLP and embeddings
sentence-transformers  # for Instructor-XL / local embeddings
numpy
pandas
pyyaml
tqdm
tenacity               # retry logic
tiktoken               # token counting
```

---

### Milestone 1: Phase 1 — Stage 1 (Summarization)
**Priority:** 🔴 Critical  
**Estimated effort:** 1–2 days

| Task | File(s) | Details |
|------|---------|---------|
| Implement summarization prompt | `src/prompts/templates.py` | Template with use-case instruction slot and target length |
| Build summarizer module | `src/phase1/summarizer.py` | Async batch summarization with configurable concurrency |
| Add format & language validation | `src/phase1/summarizer.py` | Validate output format (XML tags), detect non-English if required |
| Implement data sampling | `src/data/sampler.py` | Random sampling with configurable size (~10k for Phase 1) |
| Handle failures gracefully | `src/phase1/summarizer.py` | Discard malformed outputs (< 0.01% failure rate expected) |

**Summarization Prompt Template (from paper):**
```
You are given a conversation between a user and an AI assistant.
Summarize this conversation in {target_length} words or less,
focusing on {use_case_instruction}.

Output your summary in the following XML format:
<summary>{your summary here}</summary>
```

**Key design decisions:**
- Use async batching (e.g., 50 concurrent requests) for throughput
- Use the cheaper model (GPT-3.5-Turbo / Gemini Flash) here — quality is sufficient
- Store summaries as JSONL: `{id, original_text, summary}`

---

### Milestone 2: Phase 1 — Stage 2 (Taxonomy Generation & SGD Loop)
**Priority:** 🔴 Critical  
**Estimated effort:** 3–4 days

| Task | File(s) | Details |
|------|---------|---------|
| Implement generation prompt | `src/prompts/templates.py`, `src/phase1/taxonomy_generator.py` | Takes first batch of summaries + use-case instruction → initial taxonomy |
| Implement update prompt | `src/prompts/templates.py`, `src/phase1/taxonomy_updater.py` | Takes current taxonomy + new batch → evaluate → identify issues → update |
| Implement review prompt | `src/prompts/templates.py`, `src/phase1/taxonomy_reviewer.py` | Final quality/formatting check on the taxonomy |
| Build SGD-style iteration loop | `src/phase1/pipeline.py` | Batch summaries → chain generation → N update steps → review |
| Implement validation-based selection | `src/phase1/taxonomy_evaluator.py` | Pairwise/multi taxonomy comparison prompt on validation set |
| Multi-trial execution | `src/phase1/pipeline.py` | Run K trials (default 10), select best via validation evaluator |
| Taxonomy data model | `src/utils/parsers.py` | Dataclass/Pydantic model: `Taxonomy(labels: List[Label])` where `Label(name, description)` |

**Generation Prompt Template (from paper):**
```
You are an expert at organizing information.
Given the following {batch_size} text summaries from a corpus,
generate a label taxonomy that can be used to {use_case_instruction}.

Requirements:
- Generate exactly {num_labels} labels
- Each label should have a short name (2-5 words) and a brief description (1-2 sentences)
- Labels should be mutually exclusive and collectively exhaustive
- Labels should be at a consistent level of granularity

Summaries:
{batch_of_summaries}

Output the taxonomy in this XML format:
<taxonomy>
  <label>
    <name>{label name}</name>
    <description>{label description}</description>
  </label>
  ...
</taxonomy>
```

**Update Prompt Template (from paper):**
```
You are refining a label taxonomy for the use case: {use_case_instruction}.

Current taxonomy:
{current_taxonomy_xml}

New batch of text summaries:
{batch_of_summaries}

Please:
1. EVALUATE: Assess how well the current taxonomy fits the new data
2. IDENTIFY: List any issues (missing categories, overlapping labels, irrelevant labels)
3. UPDATE: Modify the taxonomy to address the issues

Constraints:
- Maintain exactly {num_labels} labels
- Keep label names concise (2-5 words)
- Ensure mutual exclusivity and collective exhaustiveness

Output the updated taxonomy in XML format:
<taxonomy>...</taxonomy>
```

**Key design decisions:**
- Batch size = 200 (paper default)
- Use the strong model (GPT-4 / Gemini Pro) here — reasoning quality matters
- Temperature: 0.5 for generation, 0.2 for update
- Track intermediate taxonomies for debugging and rollback
- Serialize each taxonomy version to `data/taxonomies/trial_{k}_step_{m}.json`

---

### Milestone 3: Phase 2 — LLM Label Assignment
**Priority:** 🔴 Critical  
**Estimated effort:** 2 days

| Task | File(s) | Details |
|------|---------|---------|
| Implement label assignment prompt | `src/prompts/templates.py`, `src/phase2/label_assigner.py` | Given text + full taxonomy → assign primary label + all applicable labels |
| Build async batch labeling | `src/phase2/label_assigner.py` | Process medium-to-large corpus sample with concurrent API calls |
| Output validation | `src/phase2/label_assigner.py` | Verify assigned labels exist in taxonomy, handle "Other" gracefully |
| Data splitting | `src/data/splitter.py` | Split pseudo-labeled data into train/val/test |

**Label Assignment Prompt Template (from paper):**
```
You are a text classifier. Given the following text and label taxonomy,
assign the most appropriate labels.

Text:
{document_text}

Label Taxonomy:
{taxonomy_with_descriptions}

Instructions:
1. Select the PRIMARY label that best describes this text
2. Select ALL other applicable labels (if any)
3. If no label fits well, select "Other"

Output in this format:
<assignment>
  <primary>{primary label name}</primary>
  <secondary>{comma-separated secondary labels}</secondary>
</assignment>
```

**Key design decisions:**
- Sample size for Phase 2: start with ~5k–10k, scale up if needed
- Use the strong model for labeling quality
- Store as JSONL: `{id, text, primary_label, all_labels, raw_llm_response}`

---

### Milestone 4: Phase 2 — Embedding & Classifier Training
**Priority:** 🟡 High  
**Estimated effort:** 2–3 days

| Task | File(s) | Details |
|------|---------|---------|
| Build embedding extractor | `src/phase2/embedder.py` | Support ada2 (OpenAI API), Instructor-XL (local), sentence-transformers |
| Implement LogReg classifier | `src/phase2/classifiers.py` | sklearn LogisticRegression with ℓ₂, λ ∈ {0.01, 0.1, 1, 10} |
| Implement LightGBM classifier | `src/phase2/classifiers.py` | LightGBM with max_depth ∈ {3, 5, 7, 9}, 31 leaves |
| Implement MLP classifier | `src/phase2/classifiers.py` | 2-layer MLP, Adam, lr=0.001, hidden ∈ {32, 64, 128, 256}, weight_decay=1e-5 |
| Build training pipeline with grid search | `src/phase2/trainer.py` | Train all classifiers, evaluate on val set, select best hyperparams |
| Multilabel support | `src/phase2/classifiers.py` | One-vs-all scheme for multilabel classification |
| Model persistence | `src/phase2/trainer.py` | Save/load trained models to `models/` directory |

**Key design decisions:**
- Embedding dimensions: ada2 = 1536, Instructor-XL = 768
- Grid search on validation accuracy
- For multilabel: train separate binary classifiers per label (one-vs-all)
- Save best model + hyperparams + metrics as a bundle

---

### Milestone 5: Evaluation Suite
**Priority:** 🟡 High  
**Estimated effort:** 2 days

| Task | File(s) | Details |
|------|---------|---------|
| Taxonomy coverage metric | `src/evaluation/taxonomy_metrics.py` | % of samples not assigned to "Other" |
| Pairwise label accuracy | `src/evaluation/taxonomy_metrics.py` | Generate pairs (correct label, random negative), ask rater to choose |
| Use-case relevance metric | `src/evaluation/taxonomy_metrics.py` | Binary: is this label relevant to the use-case instruction? |
| LLM-as-judge evaluator | `src/evaluation/llm_evaluator.py` | GPT-4/Gemini evaluates taxonomy quality with randomized positions |
| Classification metrics | `src/evaluation/classification_metrics.py` | Accuracy, macro/weighted P/R/F1, Cohen's κ, Fleiss' κ |
| Comparison reports | `scripts/evaluate.py` | Generate side-by-side comparison tables |

---

### Milestone 6: End-to-End Pipeline & CLI
**Priority:** 🟢 Medium  
**Estimated effort:** 1–2 days

| Task | File(s) | Details |
|------|---------|---------|
| Phase 1 CLI | `scripts/run_phase1.py` | `python run_phase1.py --config config/settings.yaml --use-case intent` |
| Phase 2 CLI | `scripts/run_phase2.py` | `python run_phase2.py --taxonomy data/taxonomies/best.json --corpus data/raw/` |
| Full pipeline CLI | `scripts/run_full_pipeline.py` | Runs Phase 1 → Phase 2 → Evaluation in sequence |
| Progress tracking | `src/utils/logging.py` | Structured JSON logging with step-level progress |
| Checkpointing | `src/phase1/pipeline.py`, `src/phase2/pipeline.py` | Resume from last completed step on failure |

---

### Milestone 7: Hierarchical Taxonomy Extension
**Priority:** 🟢 Medium  
**Estimated effort:** 1–2 days

| Task | File(s) | Details |
|------|---------|---------|
| Subgroup splitting | `src/phase1/pipeline.py` | After Phase 1 round 1, split corpus by primary label |
| Recursive Phase 1 | `src/phase1/pipeline.py` | Re-run Stage 2 on each subgroup for finer-grained labels |
| Hierarchical taxonomy model | `src/utils/parsers.py` | Tree-structured taxonomy: `Label(name, desc, children: List[Label])` |
| Hierarchical classification | `src/phase2/classifiers.py` | Top-down classification: predict L1 → predict L2 within L1 |

---

### Milestone 8: Testing & Documentation
**Priority:** 🟢 Medium  
**Estimated effort:** 1–2 days

| Task | File(s) | Details |
|------|---------|---------|
| Unit tests for parsers | `tests/test_parsers.py` | Test XML/JSON parsing, malformed input handling |
| Unit tests for prompt builders | `tests/test_taxonomy_generator.py` | Verify prompt construction with various configs |
| Integration tests | `tests/test_*.py` | Mock LLM calls, test full pipeline flow |
| README | `README.md` | Setup, usage, examples, architecture overview |

---

## Configuration Schema

```yaml
# config/settings.yaml

project:
  name: "tnt-llm-taxonomy"
  output_dir: "./data"
  models_dir: "./models"

llm:
  provider: "openai"  # openai | google | local
  summarization_model: "gpt-3.5-turbo"  # cost-efficient model
  generation_model: "gpt-4"              # high-quality model
  api_key_env: "OPENAI_API_KEY"
  max_retries: 3
  rate_limit_rpm: 60

phase1:
  sample_size: 10000
  val_ratio: 0.2
  batch_size: 200
  num_trials: 10
  summary_target_length: 20  # words
  temperatures:
    generation: 0.5
    update: 0.2
    review: 0.0
    evaluation: 0.0
  taxonomy:
    num_labels: 10           # configurable per use-case
    max_label_words: 5
    max_desc_words: 50

phase2:
  sample_size: 10000
  val_ratio: 0.15
  test_ratio: 0.15
  embedding_model: "text-embedding-ada-002"
  classifiers:
    - type: "logistic_regression"
      params:
        C: [0.01, 0.1, 1, 10]
    - type: "lightgbm"
      params:
        max_depth: [3, 5, 7, 9]
        num_leaves: 31
    - type: "mlp"
      params:
        hidden_size: [32, 64, 128, 256]
        learning_rate: 0.001
        weight_decay: 0.00001

use_case:
  name: "intent_detection"
  instruction: "Understand and categorize the primary intent of the user in this conversation"
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW TEXT CORPUS                          │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼ sample ~10k
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — STAGE 1: SUMMARIZATION                              │
│  Input:  Raw documents + use-case instruction                   │
│  Model:  GPT-3.5-Turbo / Gemini Flash (cheap, parallel)        │
│  Output: JSONL of {id, text, summary}                          │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼ batch into minibatches of 200
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — STAGE 2: TAXONOMY GENERATION (SGD LOOP)             │
│                                                                 │
│  Batch 1 ──→ [GENERATE] ──→ Taxonomy_v0                       │
│  Batch 2 ──→ [UPDATE]   ──→ Taxonomy_v1                       │
│  Batch 3 ──→ [UPDATE]   ──→ Taxonomy_v2                       │
│  ...                                                            │
│  Batch N ──→ [UPDATE]   ──→ Taxonomy_vN ──→ [REVIEW] ──→ Final│
│                                                                 │
│  × 10 trials → [VALIDATION EVAL] → Select best taxonomy       │
│  Model: GPT-4 / Gemini Pro (strong reasoning)                  │
│  Output: JSON taxonomy {labels: [{name, description}]}         │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼ best taxonomy
┌─────────────────────────────────────────────────────────────────┐
│  (OPTIONAL) HUMAN CALIBRATION                                   │
│  Light human review to refine label names & descriptions       │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼ sample ~10k (larger, different sample)
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — LABEL ASSIGNMENT (PSEUDO-LABELING)                  │
│  Input:  Raw documents + finalized taxonomy                     │
│  Model:  GPT-4 / Gemini Pro                                    │
│  Output: JSONL of {id, text, primary_label, all_labels}        │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼ embed documents
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — EMBEDDING & CLASSIFIER TRAINING                    │
│  Embeddings: ada2 (1536d) or Instructor-XL (768d)             │
│  Classifiers: LogReg, LightGBM, MLP (grid search)             │
│  Output: Trained model saved to models/                        │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼ deploy
┌─────────────────────────────────────────────────────────────────┐
│  INFERENCE (AT SCALE)                                           │
│  Offline: Batch-label entire corpus                             │
│  Online:  Real-time classification via lightweight model        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Estimated Timeline

| Milestone | Effort | Cumulative |
|-----------|--------|------------|
| M0: Setup & Infrastructure | 1 day | 1 day |
| M1: Summarization (Phase 1, Stage 1) | 1–2 days | 2–3 days |
| M2: Taxonomy Generation SGD Loop (Phase 1, Stage 2) | 3–4 days | 5–7 days |
| M3: LLM Label Assignment (Phase 2) | 2 days | 7–9 days |
| M4: Embedding & Classifier Training (Phase 2) | 2–3 days | 9–12 days |
| M5: Evaluation Suite | 2 days | 11–14 days |
| M6: End-to-End Pipeline & CLI | 1–2 days | 12–16 days |
| M7: Hierarchical Extension | 1–2 days | 13–18 days |
| M8: Testing & Documentation | 1–2 days | 14–20 days |

**Total: ~2–3 weeks for a complete implementation**

---

## LLM Provider Flexibility

The implementation should be **provider-agnostic**. The `llm_client.py` abstraction should support:

| Provider | Summarization Model | Generation/Reasoning Model | Embedding Model |
|----------|--------------------|-----------------------------|-----------------|
| OpenAI | gpt-3.5-turbo / gpt-4o-mini | gpt-4 / gpt-4o | text-embedding-ada-002 / text-embedding-3-small |
| Google | gemini-2.0-flash | gemini-2.5-pro | text-embedding-004 |
| Local | Llama 3.x (via vLLM/Ollama) | Llama 3.x 70B+ | Instructor-XL / BGE |

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM output parsing failures | Pipeline breaks | Robust XML/JSON parser with fallbacks; retry with reformatted prompt |
| API rate limits | Slow throughput | Async batching with rate limiter (tenacity + semaphore) |
| Taxonomy quality variance across trials | Inconsistent results | Run 10+ trials with validation-based selection |
| Cost of GPT-4 for large datasets | Budget overrun | Use GPT-3.5/Flash for summarization; only use strong model where needed |
| Position bias in LLM evaluation | Biased selection | Randomize option positions, average over multiple runs |
| Taxonomy size constraint violations | LLM exceeds label limit | Enforce in prompt + post-process (merge/trim excess labels) |
