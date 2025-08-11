# TnT-LLM: Text Mining at Scale with Large Language Models — Context Document

## Paper Overview

**Title:** TnT-LLM: Text Mining at Scale with Large Language Models  
**Authors:** Mengting Wan, Tara Safavi, Sujay Kumar Jauhar, Yujin Kim, Scott Counts, Jennifer Neville, Siddharth Suri, Chirag Shah, Ryen W. White, Longqi Yang, Reid Andersen, Georg Buscher, Dhruv Joshi, Nagu Rangan  
**Affiliation:** Microsoft Corporation / University of Washington  
**Source PDF:** [tntLargescaltaxonomydesign.pdf](file:///home/immortal/Desktop/Taxonomy_generation/tntLargescaltaxonomydesign.pdf)

---

## Problem Statement

Transforming unstructured text into structured, meaningful categories is a fundamental step in text mining. Current approaches suffer from:

1. **Manual taxonomy design** is expensive, time-consuming, and requires deep domain expertise
2. **Automatic clustering** (topic models, k-means) scales better but produces labels that are hard to interpret — famously likened to "reading tea leaves"
3. **Existing methods** either require predefined seed labels, or produce extractive labels (pulled verbatim from text) that lack abstraction and clarity
4. **The two tasks** — taxonomy generation and text classification — are typically treated as isolated problems, not as an end-to-end pipeline

---

## Core Idea: TnT-LLM Framework

TnT-LLM is a **two-phase, end-to-end framework** that uses LLMs to:
1. **Generate** a label taxonomy from a text corpus in a zero-shot, iterative manner
2. **Classify** text at scale by distilling LLM annotations into lightweight classifiers

The key insight is a formal analogy between **taxonomy generation** and **mixture model clustering with stochastic gradient descent (SGD)**.

---

## Phase 1: LLM-Powered Taxonomy Generation

### Stage 1 — Summarization (Feature Elicitation)

- Each document in a **small-to-medium sample** (~10k documents) is summarized by an LLM
- The prompt includes a **use-case instruction** (e.g., "summarize this conversation to understand user intent") and a **target length** (e.g., 20 words)
- This normalizes variable-length documents and extracts the **most salient signal** for the specific use case
- Analogy: this is like the **featurization step** in classical ML (projecting raw text to a representation space)
- Can be parallelized and run with a cost-efficient model (e.g., GPT-3.5-Turbo)

### Stage 2 — Taxonomy Creation, Update, and Review (Stochastic Optimization)

The summaries are divided into **minibatches** (batch size = 200 in the paper). Three prompt types are chained:

#### 2a. Generation Prompt (Initialization)
- Takes the **first minibatch** of summaries + use-case instruction
- Produces an **initial label taxonomy** (label names + descriptions)
- Analogy: Θ₀ initialization in SGD

#### 2b. Update Prompt (Iterative Refinement)
- Takes the **current taxonomy + next minibatch** of summaries
- Performs three sub-tasks per step:
  1. **Evaluate** the current taxonomy against the new data
  2. **Identify issues and suggestions** based on the evaluation
  3. **Modify** the taxonomy accordingly (add/remove/merge/rename labels)
- Analogy: SGD update step — `Θ_{m+1} = Θ_m - η∇L(Θ_m)`
- The LLM implicitly acts as both the loss function and the optimizer

#### 2c. Review Prompt (Final Polish)
- After all update steps, a review prompt checks formatting and quality
- Outputs the final taxonomy

### Validation & Model Selection
- A separate **validation set** is used with an **evaluation prompt** that compares taxonomies pairwise
- Positions are randomized to mitigate LLM position bias
- **Early stopping** can be applied — track the best taxonomy across update epochs
- Run multiple trials (10 in the paper) and select the best via validation

### Hyperparameters
- **Batch size:** 200 summaries per minibatch
- **Temperature:** 0.5 for generation prompt, 0.2 for update prompt, 0 for all others
- **Sample size:** ~10k documents for Phase 1
- **Taxonomy size:** configurable (paper used 10 intent labels, 25 domain labels)
- **Epochs:** 10 trials, each with multiple batched update steps

### Hierarchy Support
After a first round of taxonomy generation, Stage 2 can be **re-run on each subgroup** of categorized samples to create finer-grained levels → naturally supports hierarchical taxonomies.

---

## Phase 2: LLM-Augmented Text Classification

### Label Assignment (Pseudo-Labeling)
- Use the taxonomy from Phase 1 as a prompt input
- LLM classifies each document in a **medium-to-large sample** as:
  - **Primary label** (multiclass)
  - **All applicable labels** (multilabel)
- These are treated as **pseudo-labels** for training

### Classifier Distillation
- Extract **embeddings** for each document using a model (e.g., `text-embedding-ada-002`, `Instructor-XL`)
- Train **lightweight classifiers** on the pseudo-labeled data:
  - **Logistic Regression** (with ℓ₂ regularization)
  - **LightGBM** (gradient boosting)
  - **MLP** (2-layer, Adam optimizer)
- These classifiers are then deployed for **offline batch labeling** and/or **online real-time inference**

### Why Distill?
- LLM-as-classifier is accurate but slow and expensive at scale
- Distilled classifiers achieve **comparable accuracy** to GPT-4 direct classification
- Orders of magnitude **cheaper and faster** to run
- More **transparent** and debuggable

---

## Evaluation Suite

### Phase 1 Metrics (Taxonomy Quality)
| Metric | Description |
|--------|-------------|
| **Coverage** | % of samples that can be assigned a label (not "Other/Undefined") |
| **Label Accuracy** | Pairwise comparison: given text + assigned label + random negative label, does a rater pick the correct one? |
| **Use-case Relevance** | Binary rating: is the assigned label relevant to the use-case instruction? |

### Phase 2 Metrics (Classification Performance)
| Metric | Description |
|--------|-------------|
| **Inter-rater Agreement** | Cohen's κ (pairwise) and Fleiss' κ (multi-rater) |
| **Classification Metrics** | Accuracy, Macro/Weighted Precision, Recall, F1 |

### Evaluation Sources
1. **Deterministic automatic evaluation** — standard ML metrics against gold-standard annotations
2. **Human evaluation** — expert raters evaluate taxonomy quality
3. **LLM-based evaluation** — GPT-4 as an automated evaluator (validated against human agreement)

---

## Key Results

1. **TnT-LLM (GPT-4)** produces the most accurate and relevant taxonomies across all methods
2. **GPT-4 as evaluator** agrees with human majority more than humans agree with each other (Cohen's κ = 0.558–0.695 on primary label)
3. **Distilled classifiers** (LogReg/MLP on ada2 embeddings) achieve **comparable or slightly better** performance than GPT-4 direct classification on human-annotated test sets
4. **Coverage** >99.5% on both intent and domain taxonomies
5. **GPT-3.5-Turbo** works well for domain/topic labels but struggles with intent labels requiring deeper reasoning
6. **ada2 embeddings + Logistic Regression** is the sweet spot for cost-effective classification

---

## Practical Insights from the Paper

1. **Summarization is critical** for use-cases where the label space isn't evident from surface semantics (e.g., intent detection)
2. **GPT-4 >> GPT-3.5-Turbo** for taxonomy generation, especially for intent-based use cases
3. **GPT-3.5-Turbo** is sufficient for the summarization stage (Stage 1)
4. **Multiple trials + validation-based selection** significantly improves taxonomy quality
5. **Position bias** in LLM evaluation must be mitigated via randomization
6. **Lightweight human calibration** after Phase 1 improves taxonomy clarity
7. **Nonlinearity (MLP)** helps more for multilabel classification than for primary label classification

---

## Taxonomy Examples from the Paper

### User Intent Taxonomy (10 labels)
1. Fact-Based Information Seeking
2. Clarification and Concept Explanation
3. General Solution and Advice Seeking
4. Technical Assistance and Problem Solving
5. Language Translation Requests
6. Content Creation and Storytelling Requests
7. Planning and Scheduling
8. Data Analysis and Calculation Requests
9. Greetings and Social Interactions

### Conversation Domain Taxonomy (25 labels)
Academic Resources, Linguistics and Language Learning, Mathematics/Logics/Data Science, Physics and Chemistry, Business and Industry, Economics and Finance, Job and Career Advice, Legal and Regulatory, Art/Design/Creativity, Entertainment/Media/Gaming, Interactive Activities with AI, Personal Lifestyle and Hobbies, Sports and Fitness, Food and Nutrition, Health and Wellness, General Digital Support, Software Development and Hardware, Home and Household, Animals and Nature, Geography/Climate/Environment, History and Culture, Personal Counseling, Social and Political Issues, Product and Shopping, Travel and Tourism

---

## Technology Stack Used in the Paper

| Component | Technology |
|-----------|-----------|
| LLM (High-quality) | GPT-4 (0613, 32k context) |
| LLM (Cost-efficient) | GPT-3.5-Turbo (0613, 16k context) |
| Embeddings | text-embedding-ada-002, Instructor-XL |
| Classifiers | Logistic Regression, LightGBM, 2-layer MLP |
| Optimization | Grid search on validation set |
| Clustering baseline | MiniBatch K-Means with K-Means++ init |
