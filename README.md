# Taxonomy Generation from Open-Domain Questions

Turn a pile of unstructured questions into a **readable, reusable topic map** — then stamp every question with a label you can search, route, and analyze.

This repo implements the **TnT-LLM** idea from Wan et al. (*Text Mining at Scale with Large Language Models*): treat taxonomy design like clustering with SGD. Summarize the text, propose labels from the first minibatch, **update those labels as new batches arrive**, run a few independent trials, and keep the winner on a held-out set. After that, an LLM assigns each question to the winning taxonomy.

It is built around **Natural Questions (EfficientQA / NQ-open)**: short factual queries people actually type, not tidy Wikipedia abstracts. The point is to discover *specific* knowledge domains (“Constitutional Law & Legislative Procedure”), not buckets like “Science” or “History”.

---

## Why this exists

Manual taxonomies need experts and weeks. Unsupervised clustering scales, but the “topics” are often unusable. TnT-LLM sits in the middle:

1. **You state the use case** (here: granular knowledge domains in user questions).
2. **The model invents the label set** from the data, then revises it.
3. **You get two artifacts you can ship**: a taxonomy JSON, and a labeled JSONL.

Those artifacts are the product. Downstream you can power search facets, support routing, analytics dashboards, or train a cheap classifier so you stop calling an LLM for every new question.

---

## What this project actually does

| Stage | What happens | What you get |
| --- | --- | --- |
| Load & sample | JSONL/CSV/JSON in, deterministic sample + 80/20 train/val | Reproducible split (`random_seed: 42`) |
| Stage 1 — Summarize | Each question is compressed into a short *domain* summary | `output/summaries.jsonl` |
| Stage 2 — Generate | First minibatch → initial taxonomy of *N* labels | In-memory draft |
| Stage 2 — Update | Each later minibatch diagnoses gaps and rewrites labels (SGD-style) | Refined taxonomy |
| Stage 2 — Review | Format and quality polish | `output/taxonomies/trial_k.json` |
| Select | Validation summaries pick the best trial | `output/taxonomies/best_taxonomy.json` |
| Phase 2 — Assign | LLM maps each question to one primary label | `output/labeled_questions.jsonl` |

**In this run (default `config.yaml`):** 100 questions from ~1,800 NQ-open items, 10 labels, minibatch size 40, 2 trials, Groq **Qwen 3.8-27B** with **GPT-OSS-20B** fallback, rate limits baked in so the account does not get 429’d.

Not in this repo (paper Phase 2 distillation): embedding + LogReg/LightGBM/MLP. The labeled JSONL is exactly the training file you would use for that next step.

---

## Worked example (real questions from this corpus)

**Raw user questions**

```text
nucleic acids such as dna and rna store
who sings ain't nothing but a good time
which branch of the us government makes federal laws
where does the first episode of the walking dead take place
who has the longest rivalry in college football
```

**Stage 1 turns them into domain-focused summaries** (not generic “biology / music / politics”):

```text
Biology, molecular genetics, nucleic acid function, DNA and RNA information storage.
…
US government structure and legislative branch functions
Television production, specifically the filming location of a popular TV series pilot episode.
College football history and team rivalries.
```

**Stage 2 invents a 10-label map** (excerpt from the current `best_taxonomy.json`):

| Label | When a question belongs here |
| --- | --- |
| Cellular Physiology & Clinical Nomenclature | Intracellular mechanisms, genetics, medical terminology |
| Musical Attribution & Release Chronology | Who wrote/sang it, when it dropped |
| Constitutional Law & Legislative Procedure | Statutes, branches of government, legal process |
| Scripted Television Narrative & Casting | Plot, characters, actor–role mappings |
| Media Production Logistics & Distribution | Filming locations, release timelines, production ops |
| Sports Statistics & Competitive Histories | Championships, rivalries, match facts |

**After Phase 2 you can treat the corpus like a table**

| Question | Label you would use in an app |
| --- | --- |
| nucleic acids such as dna and rna store | Cellular Physiology & Clinical Nomenclature |
| who sings ain't nothing but a good time | Musical Attribution & Release Chronology |
| which branch of the us government makes federal laws | Constitutional Law & Legislative Procedure |
| where does the first episode of the walking dead take place | Media Production Logistics & Distribution |
| who has the longest rivalry in college football | Sports Statistics & Competitive Histories |

That is the whole product in miniature: **messy questions in → named shelves out**.

---

## How you would actually use it

### 1. Faceted search / “browse by topic”

Index `labeled_questions.jsonl`. A search for *legislative* can filter on `Constitutional Law & Legislative Procedure` instead of hoping keyword overlap is enough.

```python
import json

by_label = {}
with open("output/labeled_questions.jsonl") as f:
    for line in f:
        row = json.loads(line)
        key = row.get("label") or row.get("primary_label")
        by_label.setdefault(key, []).append(row["question"])

for q in by_label.get("Sports Statistics & Competitive Histories", [])[:5]:
    print("-", q)
```

### 2. Routing before you answer

A QA or RAG system can pick a retriever, prompt, or specialist model per label:

```text
Cellular Physiology…     → medical / biology corpus
Constitutional Law…      → legal / civics corpus
Scripted Television…     → entertainment knowledge base
```

You are not classifying “is this science?” You are sending *meiosis* questions to a different stack than *World Series* questions.

### 3. Analytics on what people ask

Count labels over time. If `Musical Attribution & Release Chronology` spikes, you know the traffic mix shifted — without reading a thousand logs.

### 4. Train a cheap classifier later

Use the JSONL as pseudo-labels: embed the question, train logistic regression, drop the LLM from the hot path. That is the paper’s scale story; this repo stops at high-quality labels.

### 5. Point it at *your* data

Swap the corpus path and the use-case instruction. Same pipeline works for support tickets, app reviews, or chat logs — as long as each row has a text field.

```yaml
corpus:
  path: "dataset/my_tickets.jsonl"
  text_field: "body"
  sample_size: 500

phase1:
  taxonomy:
    num_labels: 12
  use_case:
    instruction: >
      Discover a taxonomy of customer intent (billing, outage, account access, …).
      Prefer specific intents over generic buckets like "Complaint".
```

---

## Quick start

**1. Environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install openai python-dotenv pyyaml
```

**2. Secrets** (never commit `.env`)

```bash
cp .env.example .env
# GROQ_API_KEY=...
```

**3. Run**

```bash
python run_pipeline.py --config config.yaml
```

Useful flags:

```bash
python run_pipeline.py --sample-size 50          # cheaper smoke test
python run_pipeline.py --skip-labeling           # taxonomy only
python run_pipeline.py --fresh-taxonomy          # ignore cached trial_*.json
```

`tnt_pipeline.py` is a thin alias for the same entrypoint.

On reruns, summaries, taxonomy trials, and labeled rows **resume from disk** so a crash does not restart the whole Groq bill.

---

## Pipeline (mental model)

```text
NQ-open JSONL
     │
     ▼
 sample 100  ──►  train (SGD updates)  +  val (pick winner)
     │
     ▼
 summarize each question  (Stage 1)
     │
     ▼
 batch 1 → generate taxonomy
 batch 2…n → update (evaluate / suggest / rewrite)
     │
     ▼
 review → trial_1.json, trial_2.json
     │
     ▼
 LLM pairwise-style selection on val  →  best_taxonomy.json
     │
     ▼
 assign every sampled question  →  labeled_questions.jsonl
```

Prompts live in `config/prompts.yaml` (summarize, generate, update, review, evaluate, assign). Change wording there without touching Python.

---

## Configuration that matters

File: `config.yaml`

| Knob | Default here | Effect |
| --- | --- | --- |
| `corpus.sample_size` | 100 | How many questions enter the pipeline |
| `corpus.val_ratio` | 0.2 | Held out for choosing among trials |
| `phase1.taxonomy.num_labels` | 10 | Size of the taxonomy |
| `phase1.taxonomy.minibatch_size` | 40 | SGD step size (paper used 200 on larger samples) |
| `phase1.taxonomy.num_trials` | 2 | Independent generations; best wins |
| `llm.primary.model` | `qwen/qwen3.8-27b` | Generation / update quality |
| `llm.rate_limits.*` | 30 RPM-style delays | Stay under Groq quotas |

Primary model temperatures follow the paper’s spirit: higher for generation, lower for updates, near-zero for review and labeling.

---

## Repository layout

```text
config.yaml              # corpus, models, rates, use case
config/prompts.yaml      # all LLM templates
run_pipeline.py          # CLI orchestrator
llm_router.py            # Groq/OpenAI-compatible client
src/data/                # load JSONL/CSV/JSON, sample + split
src/phase1/              # summarize → generate → update → review → select
src/phase2/              # pseudo-label assignment
src/utils/               # rate limit, parse JSON, telemetry
dataset/                 # NQ-open EfficientQA questions (JSONL)
output/
  summaries.jsonl
  taxonomies/best_taxonomy.json
  labeled_questions.jsonl
```

`taxonomy/` is a local Python environment and is gitignored.

---

## Outputs to open first

1. **`output/taxonomies/best_taxonomy.json`** — the label dictionary (name + description). This is what you show a product team.
2. **`output/labeled_questions.jsonl`** — one JSON object per question: text, assigned label, confidence, short reason.
3. **`output/summaries.jsonl`** — useful if you want to debug *why* a label exists.
4. **`output/logs/`** — per-call latency, tokens, fallback (gitignored).

---

## Paper vs this implementation

Implemented: summarization, minibatch generate/update/review, multi-trial validation selection, LLM labeling, rate limiting, fallback model, checkpoint resume, externalized prompts.

Left for later: hierarchical taxonomies (re-run Stage 2 inside each label), embedding distillation, and the paper’s human/LLM evaluation suite.

The use-case prompt is deliberately **anti-generic**: the model is instructed not to emit umbrella categories. That is why the winning labels look like *Ecological Dynamics & Atmospheric Systems* rather than *Science*.

---

## Citation

Mengting Wan et al. **TnT-LLM: Text Mining at Scale with Large Language Models.** Microsoft / University of Washington.

If you change the corpus, change the use-case instruction in `config.yaml` first. The taxonomy is only as useful as the question you asked the model to answer.
