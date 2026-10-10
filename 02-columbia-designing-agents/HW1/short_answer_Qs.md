1. **[Part 2] Tokenization.** Using the output from the test function in part 2, discuss how the different input sentences
tokenized differently in terms of word-per-token and bytes-per-token. Which failed the round trip check? Why do you think that is?

**Answer:**

**Empirical Metrics (`part2_tokenizing.py` with `bpe` at `vocab_size = 500`):**
- **`tokens_per_word`** (`gpt2` / `bert` / `t5` / `bpe`):
  - `english`: `1.17` / `1.33` / `1.42` / `2.83` (equivalently `0.85` / `0.75` / `0.70` / `0.35` words-per-token)
  - `code`: `3.33` / `3.67` / `4.17` / `9.67`
  - `numbers`: `2.56` / `2.56` / `3.11` / `5.89`
  - `spanish`: `1.75` / `1.33` / `2.17` / `3.92`
  - `chinese`: `36.00` / `20.00` / `21.00` / `57.00` (no spaces in Chinese, so `len(text.split()) == 1`)
  - `emoji`: `1.46` / `1.23` / `1.77` / `3.00`
- **`bytes_per_token`** (`gpt2` / `bert` / `t5` / `bpe`):
  - `english`: `5.71` / `5.00` / `4.71` / `2.35`
  - `code`: `3.15` / `2.86` / `2.52` / `1.09`
  - `numbers`: `2.70` / `2.70` / `2.21` / `1.17`
  - `spanish`: `3.62` / `4.75` / `2.92` / `1.62`
  - `chinese`: `1.58` / `2.85` / `2.71` / `1.00`
  - `emoji`: `3.63` / `4.31` / `3.00` / `1.77`
- **Round-Trip Check (`decode(encode(text)).strip() == text.strip()`):**
  - `gpt2`: Passed all 6 (`True` on all)
  - `bpe`: Passed all 6 (`True` on all)
  - `bert` (`bert-base-multilingual-cased`): Passed `english` (`True`); **Failed** `code`, `numbers`, `spanish`, `chinese`, `emoji` (`False`)
  - `t5` (`t5-small`): Passed `english`, `code`, `numbers` (`True`); **Failed** `spanish`, `chinese`, `emoji` (`False`)

**Discussion:**
1. **Tokenization Efficiency (`tokens_per_word` and `bytes_per_token`):**
   - `english` is compressed most efficiently across all tokenizers (`1.17` tokens/word and `5.71` bytes/token in `gpt2`) because common English words and stems are stored as single tokens.
   - `code` and `numbers` have much higher `tokens_per_word` (`3.33–4.17` on pretrained; `9.67` and `5.89` on `bpe`) and low `bytes_per_token` (`1.09–3.15`) because punctuation (`<=`, `():`, `$`, `-`) and arbitrary digit runs (`1,234.56`, `2026-03-31`) split into individual symbol/digit tokens.
   - `spanish` is tokenized much more compactly by multilingual `bert` (`1.33` tokens/word, `4.75` bytes/token) than by English-centric `gpt2` (`1.75` tok/word) or `t5` (`2.17` tok/word), which fragment accented words (`rápido`, `marrón`).
   - `chinese` has no spaces (`1` "word"), so `bytes_per_token` is the fairer metric: each Chinese character is 3 UTF-8 bytes (`57` bytes total). Multilingual `bert` achieves `2.85` bytes/token (~1 token per character), whereas `gpt2` (`1.58` B/tok) splits CJK characters into raw byte pairs and our English-trained `bpe` (`1.00` B/tok) falls back to 57 individual raw bytes.
2. **Why `bert` and `t5` failed the round-trip check while `gpt2` and `bpe` succeeded:**
   - **`gpt2` and `bpe` are byte-level BPE tokenizers** whose base vocabularies include all 256 bytes (`0x00–0xFF`). Every UTF-8 sequence—even unseen Chinese characters or 4-byte emojis—is losslessly represented as byte tokens and decoded back identically.
   - **`bert` failed 5 sentences** due to: (a) **WordPiece whitespace/punctuation normalization** on `code`, `numbers`, `spanish`, and `chinese`, where `decode()` inserts spaces around punctuation (`fib ( n ) :`, `$ 1, 234. 56`, `¡ El`) and between every Chinese character (`大 语 言 模 型`); and (b) **`[UNK]` replacement** on `emoji`, where out-of-vocabulary emojis (`🚀`, `🔥`, `✨`) collapse into `[UNK]` and disappear when `skip_special_tokens=True`.
   - **`t5` failed 3 sentences (`spanish`, `chinese`, `emoji`)** because `t5-small` uses a 32k SentencePiece Unigram vocabulary without byte fallback; characters outside its Western European alphabet (`¡`, Chinese characters, and emojis) are mapped to `<unk>` and permanently lost upon decoding.

---

2. **[Part 2] Your trained BPE tokenizer.** Train your BPE tokenizer using vocab size 300, 500, and 2000. Report the vocabulary you actually got at each and account for any gap. Then tokenize the Part 2 sentences at each size. Which sentences improve as the vocabulary grows and which do not? What does that tell you about how a larger vocabulary would adapt to unseen text?

**Answer:**

1. **Actual Vocabulary Sizes & Accounting for the Gap:**
   - When trained on `BPE_CORPUS` (16 short English sentences, 151 total words, 82 unique words):
     - Requested `300` (`44` merges) $\rightarrow$ **Actual `300`**
     - Requested `500` (`244` merges) $\rightarrow$ **Actual `500`**
     - Requested `2000` (`1744` merges) $\rightarrow$ **Actual `675`** (`256` base bytes + `419` merges; **gap of 1,325 tokens**)
   - **Why the gap occurs on `BPE_CORPUS`:** Our BPE pre-tokenizes on whitespace and only merges adjacent tokens within words. After **419 merges**, every word in the 16-sentence `BPE_CORPUS` has already been reduced to a single token (`len(word) == 1`). With zero adjacent token pairs remaining (`counts` is empty), training halts early at vocab size `675`.
   - *(Note: When trained on the larger 770-review `CLEAN_REVIEWS` dataset in Part 3, all `1,744` merges complete and requested `300`, `500`, `2000` yield actual vocab sizes **`300`, `500`, and `2000`** with zero gap.)*

2. **Part 2 Sentence Metrics Across Vocab Sizes (`300` $\rightarrow$ `500` $\rightarrow$ `2000`):**
   - **On `BPE_CORPUS` (actual `300` / `500` / `675`):**
     - `english`: `4.50` $\rightarrow$ `2.83` $\rightarrow$ `2.42` tok/word (`1.48` $\rightarrow$ `2.35` $\rightarrow$ `2.76` B/tok) — **Strong improvement**
     - `emoji`: `3.77` $\rightarrow$ `3.00` $\rightarrow$ `2.69` tok/word (`1.41` $\rightarrow$ `1.77` $\rightarrow$ `1.97` B/tok) — **Moderate improvement** (English words `"Ship"`, `"it!"` merge)
     - `spanish`: `5.08` $\rightarrow$ `3.92` $\rightarrow$ `3.83` tok/word (`1.25` $\rightarrow$ `1.62` $\rightarrow$ `1.65` B/tok) — **Modest improvement** (shared Latin bigrams)
     - `numbers`: `6.22` $\rightarrow$ `5.89` $\rightarrow$ `5.89` tok/word (`1.11` $\rightarrow$ `1.17` $\rightarrow$ `1.17` B/tok) — **Plateaus after 500**
     - `code`: `9.83` $\rightarrow$ `9.67` $\rightarrow$ `9.67` tok/word (`1.07` $\rightarrow$ `1.09` $\rightarrow$ `1.09` B/tok) — **Plateaus after 500**
     - `chinese`: `57.00` $\rightarrow$ `57.00` $\rightarrow$ `57.00` tok/word (`1.00` $\rightarrow$ `1.00` $\rightarrow$ `1.00` B/tok) — **Zero improvement**
   - **On `CLEAN_REVIEWS` (actual `300` / `500` / `2000`):**
     - `english` improves from `4.25` $\rightarrow$ `2.50` $\rightarrow$ `1.17` tok/word (`1.57` $\rightarrow$ `5.71` B/tok); `spanish` (`5.17` $\rightarrow$ `2.83`) and `emoji` (`3.92` $\rightarrow$ `2.62`) improve moderately; `code` (`9.83` $\rightarrow$ `7.00`) and `numbers` (`6.22` $\rightarrow$ `5.44`) improve slightly on ASCII letters; `chinese` stays flat at `57.00` tok/word (`1.00` B/tok).

3. **Takeaway for Unseen Text:**
   - A larger vocabulary **only compresses text whose byte/character patterns appear in the tokenizer's training corpus** (`english`, and partially `spanish`/`emoji` via shared Latin letters).
   - For truly unseen scripts or modalities absent from training (`chinese` CJK bytes, `numbers`, `code` symbols), a larger vocabulary provides **no benefit at all**—the tokenizer still falls back to raw bytes (`1.00` byte/token). Thus, tokenizer coverage of unseen domains depends on **training corpus diversity**, not just vocabulary size.

---

3. **[Part 3] Training.** Report your training loss at the first and last epoch, alongside your accuracy on the test split. Account for the gap between them. What would happen to the training accuracy if we trained the model for additional epochs? Why do we measure accuracy on the test split, and not train?

**Answer:**

1. **Reported Metrics (`vocab_size = 2000`, `d_model = 64`, `epochs = 8`, `494` train / `122` test reviews):**
   - **First Epoch (Epoch 1) Training Loss:** **`0.6440`**
   - **Last Epoch (Epoch 8) Training Loss:** **`0.0028`** (trajectory: `0.6440` $\rightarrow$ `0.4198` $\rightarrow$ `0.2623` $\rightarrow$ `0.1494` $\rightarrow$ `0.0494` $\rightarrow$ `0.0163` $\rightarrow$ `0.0051` $\rightarrow$ `0.0028`)
   - **Training Split Accuracy:** **`1.000`** (`100.0%`)
   - **Test Split Accuracy:** **`0.852`** (`85.25%`; precision = `0.852` and recall = `0.852` on both positive and negative classes)

2. **Accounting for the Gap (`0.0028` train loss / `100%` train accuracy vs. `85.25%` test accuracy):**
   - The 14.75% gap between train accuracy (`100.0%`) and test accuracy (`85.25%`) is caused by **overfitting on a small dataset**: the model has **177,986 parameters** (`2000 x 64` embeddings + transformer layers) trained on only **494 reviews**. It easily memorizes specific subwords and phrasing in the training set, whereas held-out test reviews contain rarer subwords, unseen word combinations, or nuanced sentiment expressions that did not appear frequently enough in the 494 training examples.

3. **What Happens to Training Accuracy with Additional Epochs?**
   - Training accuracy has already reached its ceiling of **`1.000` (`100.0%`)**, so training for additional epochs **will keep training accuracy unchanged at `100.0%`**, while driving training cross-entropy loss even closer to `0.0000` (pushing logits toward $\pm\infty$) and potentially worsening test loss/calibration due to further overfitting.

4. **Why Measure Accuracy on the Test Split Instead of Train?**
   - High training accuracy (`100%`) merely shows that a high-capacity network can memorize its training examples. Evaluating on an independent, held-out **test split** that never influenced gradient updates measures **generalization**—how accurately the classifier predicts sentiment on new, unseen reviews in the real world.

---

4. **[Part 3] Model comparison.** Play around with the hidden state size in your transformer. Train your model and observe what happens when it's decreased to 16 and increased to 128. Compare and contrast the resulting metrics. What are the tradeoffs as model size increases?

**Answer:**

1. **Empirical Comparison (`vocab_size = 2000`, `epochs = 8`, `lr = 1e-3`, `batch_size = 32`):**

| Metric | `d_model = 16` | `d_model = 64` (Baseline) | `d_model = 128` |
| :--- | :---: | :---: | :---: |
| **Parameters** | `38,354` | `177,986` | `421,506` |
| **Training Time (8 epochs)** | `2.61 s` | `3.69 s` | `5.70 s` |
| **Epoch 1 $\rightarrow$ Epoch 8 Loss** | `0.6933` $\rightarrow$ `0.3104` | `0.6440` $\rightarrow$ `0.0028` | `0.6153` $\rightarrow$ `0.0005` |
| **Train Accuracy** | `0.901` (`90.1%`) | `1.000` (`100.0%`) | `1.000` (`100.0%`) |
| **Test Accuracy** | **`0.795`** (`79.5%`) | **`0.852`** (`85.2%`) | **`0.943`** (`94.3%`) |
| **Test Precision (Neg / Pos)** | `0.800` / `0.790` | `0.852` / `0.852` | `0.950` / `0.935` |
| **Test Recall (Neg / Pos)** | `0.787` / `0.803` | `0.852` / `0.852` | `0.934` / `0.951` |

2. **Comparison of Resulting Metrics:**
   - **Decreasing `d_model` to `16`:** Reduces parameters to `38,354` and runs fastest (`2.61s`), but severely restricts representational capacity. Loss decreases sluggishly (`0.6933 -> 0.3104`), the model cannot even fully fit the training set in 8 epochs (`90.1%` train accuracy), and test accuracy drops to **`79.5%`**.
   - **Increasing `d_model` to `128`:** Expands parameters to `421,506` (`11x` larger than `d_model=16`). Optimization converges much faster (`0.6153 -> 0.0322` by Epoch 4, `0.0005` by Epoch 8), and test accuracy rises to **`94.3%`** (`+9.0%` over `d_model=64` and `+14.8%` over `d_model=16`), with balanced ~`0.93–0.95` precision and recall across both classes.

3. **Tradeoffs as Model Size Increases:**
   - **Compute & Latency vs. Accuracy:** Doubling hidden dimension increases linear/FFN layer FLOPs quadratically ($O(L \cdot d_{\text{model}}^2)$) and attention FLOPs linearly ($O(L^2 \cdot d_{\text{model}})$), increasing training time by **`2.18x`** (`2.61s` $\rightarrow$ `5.70s`) and inference latency proportionally.
   - **Memory Footprint:** Parameter and optimizer memory grow **`11x`** (`38.4K` $\rightarrow$ `421.5K` parameters).
   - **Expressivity vs. Overfitting Risk:** Larger `d_model` provides richer token embeddings and smoother gradient flow (improving test accuracy from `79.5%` to `94.3%`), but also saturates training loss much faster (`0.0005`), increasing the risk of memorizing noise on small datasets without regularization or early stopping.
