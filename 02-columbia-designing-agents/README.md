# Columbia COMS 4995-15: Designing LLM Agents (Fall 2026) — Labs & Homeworks

Complete, verified solutions for all **In-Class Labs (`labs/`)**, **Homework 1 (`HW1/`)**, and **Homework 2 (`HW2/`)** of Columbia University's **COMS 4995-15: Designing LLM Agents** (Fall 2026, Prof. Shreya Havaldar).

---

## Directory Structure

```text
02-columbia-designing-agents/
├── README.md                              # Course overview, reproduction commands, and verification summary
├── labs/                                  # In-Class Coding Labs & Week 4 Lecture Study Guide
│   ├── gradient_descent.py                # Week 1 Lab: 4-layer MLP (784-16-16-10) on MNIST from scratch in NumPy
│   ├── beam_search.py                     # Week 3 Lab: Causal LM sequence scoring, beam search, & length normalization
│   ├── autoprompting.py                   # Week 4 Lab: Automated prompt optimization (Generator + Judge + Meta-Prompt Rewriter)
│   ├── autoprompting_zh.py                # Week 4 Lab (Bilingual Chinese annotated edition)
│   ├── gemini.py                          # Week 4 Helper: Gemini API wrapper with env key loading & offline fallback
│   ├── gemini_zh.py                       # Week 4 Helper (Bilingual Chinese annotated edition)
│   ├── week4.pdf                          # Week 4 Lecture Slides: Prompting, Reasoning & Memory (56 slides)
│   └── Week4_Prompting_Reasoning_Memory_zh.md # Full Chinese translation, study guide, & Slide 56 discussion answers
├── HW1/                                   # Homework 1: Data Processing, Tokenization & Transformer Classifier
│   ├── part1_data_processing.py           # Tasks 1.0–1.5: Deduplication, length/gibberish/repetition filtering, PII redaction
│   ├── part2_tokenizing.py                # Tasks 2.0–2.4: GPT-2/mBERT/T5 comparison & byte-level BPE tokenizer from scratch
│   ├── part3_classifier.ipynb             # Tasks 3.0–3.5: TinyTransformer sentiment classifier with executed outputs
│   ├── run_part3_experiments.py           # Standalone runner for Part 2 & Part 3 vocabulary / hidden-size experiments
│   ├── part3_experiment_results.json      # Saved metrics across vocab_size=(300,500,2000) and d_model=(16,64,128)
│   ├── short_answer_Qs.md                 # Questions 1–4 with full empirical answers
│   ├── HW1_short_answers.md               # Extended report with tables and analysis for HW1 Questions 1–4
│   └── data/
│       ├── raw_reviews.jsonl              # 1,000 raw e-commerce reviews
│       ├── processed_data.jsonl           # 770 cleaned & PII-redacted reviews produced by part1_data_processing.py
│       └── test_cases.json                # 52 unit test cases for Part 1 (52/52 passing)
└── HW2/                                   # Homework 2: Decoding, Prompting, Retrieval (RAG) & Agent Memory
    ├── part1_decoding.py                  # Tasks 1.0–1.7: Greedy, temperature, top-k, top-p, repetition penalty, distinct-n
    ├── part2_prompting.py                 # Tasks 2.0–2.5: Zero-shot, few-shot, Chain-of-Thought, & Self-Consistency extraction
    ├── part3_retrieval.py                 # Tasks 3.0–3.5: Sliding-window chunking, dense BGE vs TF-IDF, grounded QA & abstention
    ├── part4_memory.py                    # Tasks 4.0–4.4: 3-tier agent memory (FIFO Buffer + Rolling Summary + Vector FactStore)
    ├── run_hw2_experiments.py             # Standalone runner for Parts 1–4 and Q2/Q4/Q5 ablation studies
    ├── hw2_experiment_results.json        # Saved empirical results for all HW2 experiments
    ├── short_answer_Qs.md                 # Questions 1–5 with full empirical answers
    ├── HW2_short_answers.md               # Extended report with tables and analysis for HW2 Questions 1–5
    └── data/
        ├── tickets_labeled.jsonl          # Fossil Falls support tickets with gold structured extractions
        ├── help_articles.jsonl            # 30 Fossil Falls help desk knowledge-base articles (KB-001..KB-030)
        ├── retrieval_queries.jsonl        # 30 evaluation queries (12 paraphrase, 12 exact, 6 unanswerable)
        └── conversation.json              # 22-turn multi-topic guest conversation for memory evaluation
```

---

## Verification & Key Empirical Results

### 1. In-Class Labs (`labs/`)

- **`labs/gradient_descent.py`** (`python3 labs/gradient_descent.py`):
  - Implements `load_mnist`, `init_network` (Xavier $1/\sqrt{\text{fan\_in}}$), `forward` (sigmoid hidden layers + numerically stable log-sum-exp softmax), `loss` (mean cross-entropy), `numerical_gradient` (central finite differences, max diff `< 1e-8` vs. analytical backprop), `backward`, and mini-batch SGD `train`.
  - Trains for 20 epochs (`LR = 0.1`, `BATCH = 64`), reducing training loss from `0.8357` to `0.0088` (`1.000` train accuracy, `0.901` test accuracy).
- **`labs/beam_search.py`** (`python3 labs/beam_search.py`):
  - Implements `load_model`, `sequence_logprob`, `expand`, `beam_search` (`BEAM_WIDTHS = (1, 2, 4)`), `length_normalize` ($\text{logprob} / L^\alpha$ with $\alpha = 0.7$), and `best_beam`.
  - Verifies exact causal sequence scoring (`running score == recomputed` to `0.000` difference) and demonstrates how length normalization (`ALPHA = 0.7`) prevents short-beam bias (`"Jupiter."` at `-2.06` vs. `"Jupiter, the fifth planet from the Sun."` winning after length normalization).
- **`labs/autoprompting.py`** (`python3 labs/autoprompting.py`):
  - Implements iterative meta-prompt optimization (`GENERATOR_PROMPT`, `ERROR_TAXONOMY`, `judge`, `rewrite`, `autoprompt`), reducing test-suite errors across 3 rounds from `3 -> 2 -> 1 -> 0`.

---

### 2. Homework 1 (`HW1/`)

- **Part 1 — Data Processing (`python3 HW1/part1_data_processing.py`)**:
  - Passes **52 / 52** unit tests in `HW1/data/test_cases.json`.
  - Filters 1,000 raw reviews down to **770 clean reviews** in `HW1/data/processed_data.jsonl` (`55` dropped by `length`, `55` by `gibberish`, `120` by `repetition`) with all emails, phone numbers, and credit cards redacted.
- **Part 2 — Tokenization (`python3 HW1/part2_tokenizing.py`)**:
  - Evaluates `gpt2`, `bert-base-multilingual-cased`, `t5-small`, and custom byte-level BPE (`learn_merges`, `apply_merges`, `train_bpe`) across `english`, `code`, `numbers`, `spanish`, `chinese`, and `emoji`.
  - Byte-level `gpt2` and custom `bpe` pass **6 / 6** round-trip checks (`True`), whereas `bert` fails 5 / 6 (due to WordPiece punctuation/CJK spacing and `[UNK]` on emojis) and `t5` fails 3 / 6 (`<unk>` on `¡`, Chinese, and emojis).
- **Part 3 — Transformer Sentiment Classifier (`HW1/part3_classifier.ipynb` & `python3 HW1/run_part3_experiments.py`)**:
  - Baseline (`vocab_size = 2000`, `d_model = 64`, `epochs = 8`): Epoch 1 loss `0.6440` $\rightarrow$ Epoch 8 loss `0.0028`; Train accuracy `1.000`, Test accuracy **`0.852`**.
  - Hidden dimension ablation: `d_model = 16` (`38,354` params, Test Acc **`0.795`**), `d_model = 64` (`177,986` params, Test Acc **`0.852`**), `d_model = 128` (`421,506` params, Test Acc **`0.943`**).

---

### 3. Homework 2 (`HW2/`)

- **Part 1 — Decoding (`python3 HW2/part1_decoding.py`)**:
  - Compares `greedy` (`distinct-2 = 0.182`, repetitive loop), `temperature 0.7` (`0.929`), `temperature 1.5` (`1.000`, incoherent tail tokens), `top-k 5` (`0.567`), `top-p 0.9` (`0.594`), and `top-p 0.9 + penalty` (`0.964`, fluent and non-repetitive).
- **Part 2 — Structured Prompting (`python3 HW2/part2_prompting.py`)**:
  - 100% schema validity (`1.000`) across all 4 extractors; `exact_match` improves from **`0.800`** (`zero-shot`) $\rightarrow$ **`0.900`** (`few-shot`) $\rightarrow$ **`1.000`** (`cot` and `consistent`).
- **Part 3 — Retrieval & Grounded Answering (`python3 HW2/part3_retrieval.py`)**:
  - Chunks 30 articles into 91 overlapping windows (`CHUNK_SIZE = 60`, `CHUNK_OVERLAP = 15`).
  - Dense embeddings achieve **`1.000` `recall@1` on `paraphrase` queries** (vs. `0.333` for TF-IDF), while TF-IDF achieves **`0.917` `recall@1` and `1.000` `recall@3` on `exact` queries** like `GATE-4102` and `RAIL-5510` (vs. `0.583` `recall@1` for embeddings).
  - Grounded QA achieves **`1.000` true abstention** on unanswerable queries and **`0.000` false abstention** on answerable queries.
- **Part 4 — Three-Layer Agent Memory (`python3 HW2/part4_memory.py`)**:
  - Combines FIFO `Buffer`, rolling LLM `Summary` (`< 40` words), and vector-indexed `FactStore` across a 22-turn guest conversation (`175` tokens at `buffer 2`, `203` tokens at `buffer 4`, `278` tokens at `buffer 8`).
