# COMS 4995-15 (Fall 2026) — Homework 1: Short Answer Responses

---

## 1. [Part 2] Tokenization

> **Question:** Using the output from the test function in part 2, discuss how the different input sentences tokenized differently in terms of word-per-token and bytes-per-token. Which failed the round trip check? Why do you think that is?

### Empirical Results from `part2_tokenizing.py`

#### Tokens per Word (`tokens_per_word` = `len(encode(text)) / len(text.split())`, lower means fewer tokens per whitespace-delimited word; inverse of words-per-token):

| Sentence Type | `gpt2` | `bert` (`bert-base-multilingual-cased`) | `t5` (`t5-small`) | Custom `bpe` (`vocab=500`) |
| :--- | :---: | :---: | :---: | :---: |
| **`english`** | `1.17` | `1.33` | `1.42` | `2.83` |
| **`code`** | `3.33` | `3.67` | `4.17` | `9.67` |
| **`numbers`** | `2.56` | `2.56` | `3.11` | `5.89` |
| **`spanish`** | `1.75` | `1.33` | `2.17` | `3.92` |
| **`chinese`** | `36.00` | `20.00` | `21.00` | `57.00` |
| **`emoji`** | `1.46` | `1.23` | `1.77` | `3.00` |

#### Bytes per Token (`bytes_per_token` = `len(text.encode("utf-8")) / len(encode(text))`, higher means greater compression efficiency):

| Sentence Type | `gpt2` | `bert` (`bert-base-multilingual-cased`) | `t5` (`t5-small`) | Custom `bpe` (`vocab=500`) |
| :--- | :---: | :---: | :---: | :---: |
| **`english`** | `5.71` | `5.00` | `4.71` | `2.35` |
| **`code`** | `3.15` | `2.86` | `2.52` | `1.09` |
| **`numbers`** | `2.70` | `2.70` | `2.21` | `1.17` |
| **`spanish`** | `3.62` | `4.75` | `2.92` | `1.62` |
| **`chinese`** | `1.58` | `2.85` | `2.71` | `1.00` |
| **`emoji`** | `3.63` | `4.31` | `3.00` | `1.77` |

#### Round-Trip Check (`decode(encode(text)).strip() == text.strip()`):

| Sentence Type | `gpt2` | `bert` (`bert-base-multilingual-cased`) | `t5` (`t5-small`) | Custom `bpe` (`vocab=500`) |
| :--- | :---: | :---: | :---: | :---: |
| **`english`** | `True` | `True` | `True` | `True` |
| **`code`** | `True` | `False` | `True` | `True` |
| **`numbers`** | `True` | `False` | `True` | `True` |
| **`spanish`** | `True` | `False` | `False` | `True` |
| **`chinese`** | `True` | `False` | `False` | `True` |
| **`emoji`** | `True` | `False` | `False` | `True` |

### Analysis & Discussion

1. **Differences in `tokens_per_word` (and `words_per_token`) and `bytes_per_token` across inputs:**
   - **Plain English (`english`)** achieves the highest compression across all tokenizers (`1.17` tokens/word and `5.71` bytes/token on `gpt2`), because the training corpora of GPT-2, T5, and our custom BPE are predominantly English prose where whole words (`tokenization`, `shapes`, `reasoning`) are merged into single subword tokens.
   - **Code (`code`) and Numbers (`numbers`)** fragment heavily (`3.33–4.17` tokens/word on pretrained models; `9.67` on custom BPE) and drop to `2.21–3.15` bytes/token (`1.09–1.17` on custom BPE). Punctuation sequences (`def`, `fib(n):`, `<=`, `n-1`, `n-2`), formatting symbols (`$1,234.56`, `2026-03-31`), and multi-digit sequences are split into small symbol/digit chunks because arbitrary digit combinations and operators are not merged together into single vocabulary entries.
   - **Spanish (`spanish`)** shows a sharp divergence between monolingual/English-centric tokenizers (`gpt2`: `1.75` tokens/word, `t5`: `2.17` tokens/word) and multilingual BERT (`bert-base-multilingual-cased`: `1.33` tokens/word, `4.75` bytes/token). Multilingual BERT includes Spanish words and accented morphemes (`rápido`, `marrón`) in its vocabulary, whereas English-centric GPT-2 and T5 split accented words into smaller subwords.
   - **Chinese (`chinese`)** has no whitespace between words (`len(text.split()) == 1`), so `tokens_per_word` equals the total number of tokens in the entire sentence (`36.0` for `gpt2`, `20.0` for `bert`, `21.0` for `t5`, `57.0` for custom `bpe`). Looking at `bytes_per_token` reveals the true compression: each CJK character is 3 UTF-8 bytes (`57` bytes total for 19 characters). Multilingual `bert` achieves `2.85` bytes/token (~1 token per Chinese character), whereas `gpt2` (`1.58` bytes/token) splits Chinese characters into individual UTF-8 bytes or byte pairs, and our English-trained custom `bpe` (`1.00` bytes/token) falls back completely to raw single UTF-8 bytes (`57` tokens for `57` bytes).
   - **Emoji (`emoji`)** uses 4-byte UTF-8 sequences per emoji (`🚀`, `🔥`, `✨`). Pretrained byte-level `gpt2` (`1.46` tokens/word, `3.63` bytes/token) and our byte-level `bpe` (`1.77` bytes/token) split unseen emojis into raw UTF-8 byte tokens without losing information.

2. **Which failed the round-trip check, and why?**
   - **`gpt2` and our custom `bpe` passed all 6/6 round-trip checks (`True` on every sentence)** because both are **byte-level BPE tokenizers**. Their base vocabulary contains all 256 raw byte values (`0x00`–`0xFF`), guaranteeing that any valid UTF-8 string can be tokenized into byte/merged-byte tokens and losslessly reconstructed via UTF-8 decoding—even for unseen scripts (Chinese) and 4-byte emojis.
   - **`bert` (`bert-base-multilingual-cased`) failed 5 out of 6 sentences (`code`, `numbers`, `spanish`, `chinese`, `emoji`)** for two distinct architectural reasons:
     1. **Destructive whitespace normalization during WordPiece decoding (`code`, `numbers`, `spanish`, `chinese`)**: BERT's WordPiece tokenizer strips whitespace during tokenization and treats all punctuation characters as isolated tokens; when `decode(..., skip_special_tokens=True)` reconstructs the string, it inserts spaces around punctuation (`fib ( n ) :`, `$ 1, 234. 56`, `2026 - 03 - 31`, `¡ El zorro`) and between every CJK character (`大 语 言 模 型 ...`), breaking exact string equality with the original input.
     2. **Out-of-vocabulary `[UNK]` replacement (`emoji`)**: Because WordPiece operates on Unicode characters with a fixed character alphabet rather than raw bytes, unseen emojis (`🚀`, `🔥`, `✨`) are mapped to the single `[UNK]` token. Once collapsed to `[UNK]` (which is stripped when `skip_special_tokens=True`), the original characters are permanently lost (`"Ship it ! -> ship it !"`).
   - **`t5` (`t5-small`) failed 3 out of 6 sentences (`spanish`, `chinese`, `emoji`)**: `t5-small` uses a SentencePiece Unigram vocabulary trained only on English and a few Western European languages (`32,000` tokens) **without byte fallback**. Consequently, characters outside its vocabulary—such as the Spanish inverted exclamation mark `¡`, all Chinese characters (`大语言模型...`), and emojis (`🚀`, `🔥`, `✨`)—are replaced with `<unk>`. When decoded with `skip_special_tokens=True`, all `<unk>` tokens vanish, causing irreversible character loss.

---

## 2. [Part 2] Your Trained BPE Tokenizer

> **Question:** Train your BPE tokenizer using vocab size 300, 500, and 2000. Report the vocabulary you actually got at each and account for any gap. Then tokenize the Part 2 sentences at each size. Which sentences improve as the vocabulary grows and which do not? What does that tell you about how a larger vocabulary would adapt to unseen text?

### 1. Actual Vocabulary Sizes Achieved & Explanation of the Gap

We evaluated `train_bpe(corpus, vocab_size)` on both the 16-sentence `BPE_CORPUS` in `part2_tokenizing.py` and the 770-review `CLEAN_REVIEWS` corpus in `part3_classifier.ipynb`:

| Requested `vocab_size` | Base Byte Vocab | Requested Merges (`vocab_size - 256`) | Actual Vocab on `BPE_CORPUS` (16 sentences) | Actual Vocab on `CLEAN_REVIEWS` (770 reviews) |
| :---: | :---: | :---: | :---: | :---: |
| **`300`** | `256` | `44` | **`300`** (`256 + 44` merges) | **`300`** (`256 + 44` merges) |
| **`500`** | `256` | `244` | **`500`** (`256 + 244` merges) | **`500`** (`256 + 244` merges) |
| **`2000`** | `256` | `1744` | **`675`** (`256 + 419` merges — **gap of 1,325**) | **`2000`** (`256 + 1744` merges — **gap of 0**) |

- **Why there is a gap at `vocab_size = 2000` on `BPE_CORPUS` (`675` vs. `2000`):**
  - Our BPE algorithm pre-tokenizes the corpus by splitting on whitespace (`[list(w.encode("utf-8")) for line in corpus for w in line.split()]`) and never merges across word boundaries.
  - `BPE_CORPUS` is tiny (only 16 short English sentences, 151 total whitespace-delimited word tokens, and 82 unique words). After **419 merges** (`256 + 419 = 675` total vocabulary entries), **every single word in `BPE_CORPUS` has already been merged into a single 1-token sequence** (`len(word) == 1`). At step 420, there are zero adjacent token pairs left anywhere in the corpus (`not counts`), so `learn_merges` terminates early at vocabulary size **675**.
  - By contrast, when trained on the larger 770-review `CLEAN_REVIEWS` dataset in Part 3 (thousands of words), there are more than enough recurring subword pairs to complete all `1,744` merges and reach the full vocabulary size of **2,000**.

### 2. Tokenization of Part 2 Sentences Across Vocabulary Sizes (`300`, `500`, `2000`)

#### Table A: Trained on `BPE_CORPUS` (Actual Vocab: `300`, `500`, `675`)

| Sentence | `vocab=300` (tok/word \| B/tok) | `vocab=500` (tok/word \| B/tok) | `vocab=2000` [actual `675`] (tok/word \| B/tok) | Net Change (`300 -> 2000`) |
| :--- | :---: | :---: | :---: | :--- |
| **`english`** | `4.50` \| `1.48` | `2.83` \| `2.35` | `2.42` \| `2.76` | **Strong improvement** (`-46.2%` tok/word) |
| **`emoji`** | `3.77` \| `1.41` | `3.00` \| `1.77` | `2.69` \| `1.97` | **Moderate improvement** (`-28.6%` tok/word, from English words `"Ship"`, `"it!"`) |
| **`spanish`** | `5.08` \| `1.25` | `3.92` \| `1.62` | `3.83` \| `1.65` | **Modest improvement** (`-24.6%` tok/word, shared Latin bigrams like `es`, `la`, `re`) |
| **`numbers`** | `6.22` \| `1.11` | `5.89` \| `1.17` | `5.89` \| `1.17` | **Almost no improvement** (plateaus at `500`; no digits exist in `BPE_CORPUS`) |
| **`code`** | `9.83` \| `1.07` | `9.67` \| `1.09` | `9.67` \| `1.09` | **Almost no improvement** (plateaus at `500`; no code syntax in `BPE_CORPUS`) |
| **`chinese`** | `57.00` \| `1.00` | `57.00` \| `1.00` | `57.00` \| `1.00` | **Zero improvement** (`1.00` byte/token at all sizes) |

#### Table B: Trained on `CLEAN_REVIEWS` (Actual Vocab: `300`, `500`, `2000`)

| Sentence | `vocab=300` (tok/word \| B/tok) | `vocab=500` (tok/word \| B/tok) | `vocab=2000` (tok/word \| B/tok) | Net Change (`300 -> 2000`) |
| :--- | :---: | :---: | :---: | :--- |
| **`english`** | `4.25` \| `1.57` | `2.50` \| `2.67` | `1.17` \| `5.71` | **Massive improvement** (`-72.5%` tok/word; matches GPT-2 efficiency!) |
| **`spanish`** | `5.17` \| `1.23` | `3.42` \| `1.85` | `2.83` \| `2.24` | **Moderate improvement** (shared Latin character sequences) |
| **`emoji`** | `3.92` \| `1.35` | `3.15` \| `1.68` | `2.62` \| `2.03` | **Moderate improvement** (English words `"Ship"`, `"it!"` merge; emojis stay raw bytes) |
| **`code`** | `9.83` \| `1.07` | `9.33` \| `1.12` | `7.00` \| `1.50` | **Small improvement** (ASCII letters `return`, `if`, `def` merge; operators do not) |
| **`numbers`** | `6.22` \| `1.11` | `5.78` \| `1.19` | `5.44` \| `1.27` | **Very small improvement** (English words `"Order"`, `"placed"`, `"at"` merge; digits stay fragmented) |
| **`chinese`** | `57.00` \| `1.00` | `57.00` \| `1.00` | `57.00` \| `1.00` | **Zero improvement** (`1.00` byte/token at all sizes) |

### 3. Which Sentences Improve vs. Do Not, and Implications for Unseen Text

- **Which improve:**
  - **`english`** improves dramatically as vocabulary size grows (`4.50 -> 2.42` tok/word on `BPE_CORPUS`, and `4.25 -> 1.17` tok/word on `CLEAN_REVIEWS`), because the training corpus is English text and larger vocabularies merge frequent English stems, suffixes (`-ing`, `-tion`), and whole words.
  - **`spanish`** and **`emoji`** improve partially because they share Latin roots/syllables (`spanish`) or contain English words (`"Ship it!"` in `emoji`).
- **Which do not improve (or barely improve):**
  - **`chinese`** shows **zero improvement** (`57.0` tokens/word, `1.00` byte/token across `300`, `500`, and `2000`), because no CJK UTF-8 bytes (`0xE4`–`0xE9`, etc.) ever appear in the English training corpus, so zero merges are ever learned for those byte sequences.
  - **`numbers`** and **`code`** barely improve (`5.89` and `9.67` tok/word on `BPE_CORPUS`), improving only on the few English letters (`if`, `return`, `Order`, `placed`) while digits and syntax symbols remain unmerged single bytes.
- **What this tells us about how a larger vocabulary adapts to unseen text:**
  - Increasing `vocab_size` **only improves compression and token efficiency on text whose character/byte distributions overlap with the tokenizer's training corpus** (in-distribution languages and domains).
  - A larger vocabulary **cannot adapt to out-of-distribution scripts, languages, or modalities** (such as Chinese, emojis, source code, or numeric logs) if those byte patterns were absent from the BPE training data. Furthermore, over-allocating vocabulary slots to domain-specific whole-word merges in one domain leaves unseen domains fragmented into raw bytes (`1.00` byte/token), inflating sequence lengths, attention compute ($O(L^2)$), and out-of-distribution token fragmentation. Diverse, multilingual, multi-domain pre-training corpora are just as essential for tokenizer training as vocabulary size itself.

---

## 3. [Part 3] Training

> **Question:** Report your training loss at the first and last epoch, alongside your accuracy on the test split. Account for the gap between them. What would happen to the training accuracy if we trained the model for additional epochs? Why do we measure accuracy on the test split, and not train?

### 1. Reported Metrics (`vocab_size = 2000`, `d_model = 64`, `epochs = 8`, `lr = 1e-3`, `batch_size = 32`)

- **Dataset Split (after Part 1 cleaning and dropping neutral 3-star reviews, balanced):**
  - Total clean reviews: `770` $\rightarrow$ Binary labeled & balanced: `494` training examples (`247` positive, `247` negative) and `122` test examples (`61` positive, `61` negative).
- **Training Loss Trajectory Across 8 Epochs:**
  - **Epoch 1 (First Epoch) Training Loss:** **`0.6440`** (near $\ln 2 \approx 0.6931$, random guessing for binary cross-entropy)
  - Epoch 2 Loss: `0.4198`
  - Epoch 3 Loss: `0.2623`
  - Epoch 4 Loss: `0.1494`
  - Epoch 5 Loss: `0.0494`
  - Epoch 6 Loss: `0.0163`
  - Epoch 7 Loss: `0.0051`
  - **Epoch 8 (Last Epoch) Training Loss:** **`0.0028`**
- **Final Accuracy & Classification Metrics:**
  - **Training Split Accuracy:** **`1.000`** (`100.0%`; precision = `1.000`, recall = `1.000` for both classes)
  - **Test Split Accuracy:** **`0.852`** (`85.25%`; `precision_neg = 0.852`, `recall_neg = 0.852`, `precision_pos = 0.852`, `recall_pos = 0.852`)

### 2. Accounting for the Gap Between Training Performance and Test Accuracy

- By Epoch 8, the training loss is nearly zero (`0.0028`) and training accuracy reaches `100.0%`, while test accuracy is `85.25%` (a **14.75 percentage point generalization gap**).
- This gap is due to **overfitting (memorization vs. generalization)**:
  1. **High model capacity relative to dataset size:** With `vocab_size = 2000` and `d_model = 64`, the `TinyTransformer` has **177,986 trainable parameters** (mostly in the `2000 x 64 = 128,000` token embedding table and `64 x 64` attention/FFN layers), while the training set has only **494 reviews**.
  2. **Idiosyncratic token memorization & unseen test subwords:** The model can achieve near-zero loss on the 494 training examples by memorizing specific rare tokens or phrasing patterns present in the training reviews. When evaluated on the 122 held-out test reviews, some test reviews contain unseen word combinations, subtle sarcasm/mixed sentiment, or subwords that received few gradient updates during training, leading to 18 misclassified test examples out of 122 (`104 / 122 = 85.25%`).

### 3. What Would Happen to Training Accuracy with Additional Epochs?

- Since **training accuracy has already saturated at `1.000` (`100.0%`)** by Epoch 8, training for additional epochs **cannot increase training accuracy any further—it will remain at `1.000` (`100%`)** (barring rare optimizer instability).
- Meanwhile, the **training cross-entropy loss** would continue to asymptotically approach `0.0000` as the model pushes its output logits to larger magnitudes ($\pm \infty$) to become even more confident on the already-correct training examples. On the **test split**, this over-confident logit scaling would increase **test loss** (miscalibration) and eventually stagnate or slightly degrade test accuracy due to overfitting.

### 4. Why We Measure Accuracy on the Test Split Instead of the Training Split

- The goal of a machine learning classifier is **generalization to unseen data** in deployment, not rote memorization of historical examples it was already given labels for.
- A model with sufficient parameters (here ~178K parameters for 494 samples) can act like a lookup table and achieve `100%` accuracy on the training split even if it relies on spurious correlations. Measuring accuracy on a held-out **test split** that never contributed to gradient updates (nor BPE/model parameter fitting) provides an unbiased estimate of how well the learned representations transfer to new, unseen customer reviews.

---

## 4. [Part 3] Model Comparison

> **Question:** Play around with the hidden state size in your transformer. Train your model and observe what happens when it's decreased to 16 and increased to 128. Compare and contrast the resulting metrics. What are the tradeoffs as model size increases?

### 1. Empirical Comparison Across Hidden State Sizes (`d_model = 16`, `64`, `128`)

All three models were trained under identical conditions (`vocab_size = 2000`, `MAX_LEN = 128`, `EPOCHS = 8`, `LR = 1e-3`, `BATCH_SIZE = 32`, `seed = 42`):

| Metric | `d_model = 16` (Small) | `d_model = 64` (Baseline) | `d_model = 128` (Large) |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | `38,354` | `177,986` (`4.6x`) | `421,506` (`11.0x`) |
| **Wall-Clock Training Time (8 epochs)** | `2.61 s` | `3.69 s` (`1.41x`) | `5.70 s` (`2.18x`) |
| **Epoch 1 Training Loss** | `0.6933` | `0.6440` | `0.6153` |
| **Epoch 4 Training Loss** | `0.6343` | `0.1494` | `0.0322` |
| **Epoch 8 (Final) Training Loss** | `0.3104` | `0.0028` | `0.0005` |
| **Train Accuracy** | `0.901` (`90.08%`) | `1.000` (`100.0%`) | `1.000` (`100.0%`) |
| **Test Accuracy** | **`0.795`** (`79.51%`) | **`0.852`** (`85.25%`) | **`0.943`** (`94.26%`) |
| **Test Precision (Neg / Pos)** | `0.800` / `0.790` | `0.852` / `0.852` | `0.950` / `0.935` |
| **Test Recall (Neg / Pos)** | `0.787` / `0.803` | `0.852` / `0.852` | `0.934` / `0.951` |

*(Bonus reference: Varying `vocab_size` at fixed `d_model = 64` yielded test accuracies of `0.828` at `vocab=300`, `0.893` at `vocab=500`, and `0.852` at `vocab=2000`, showing that a compact `vocab=500` actually generalizes slightly better at `d_model=64` because each subword embedding receives more frequent gradient updates.)*

### 2. Comparing and Contrasting the Resulting Metrics

- **Decreasing `d_model` to `16` (Under-parameterized / Slow Convergence):**
  - The parameter count drops by `78.5%` (to `38,354`), and convergence is much slower: the training loss stays flat near random guessing (`0.6933 -> 0.6343`) for the first 4 epochs and only reaches `0.3104` by Epoch 8.
  - Because a 16-dimensional embedding and single-head 16-d attention bottleneck cannot capture rich semantic nuances across a 2,000-token vocabulary, the model underfits within 8 epochs: training accuracy only reaches `90.1%`, and test accuracy drops to **`79.5%`** (with precision/recall ~`0.79–0.80`).
- **Baseline `d_model = 64`:**
  - Converges rapidly (`0.6440 -> 0.0028` loss), reaches `100.0%` training accuracy, and achieves **`85.2%`** balanced test accuracy (`0.852` precision and recall on both classes).
- **Increasing `d_model` to `128` (Higher Representational Capacity & Faster Optimization):**
  - The parameter count increases `2.37x` over baseline (`421,506` parameters vs. `177,986`, and `11x` over `d_model=16`).
  - Training loss plunges much faster—dropping to `0.0322` by Epoch 4 and `0.0005` by Epoch 8.
  - Crucially, test accuracy jumps to **`94.3%`** (`precision_neg = 0.950, recall_neg = 0.934, precision_pos = 0.935, recall_pos = 0.951`), outperforming `d_model = 64` by **+9.0 percentage points** and `d_model = 16` by **+14.8 percentage points**. Higher-dimensional embeddings and wider feed-forward projections (`128 -> 256 -> 128`) provide a smoother optimization landscape and allow attention to linearly separate positive and negative sentiment features more cleanly.

### 3. Tradeoffs as Model Size Increases

1. **Compute Time & Latency:** Training time more than doubles from `d_model = 16` (`2.61s`) to `d_model = 128` (`5.70s`, `2.18x` slower). In a Transformer, projection and FFN layers scale quadratically with hidden dimension $O(L \cdot d_{\text{model}}^2)$, while attention score computation scales as $O(L^2 \cdot d_{\text{model}})$.
2. **Memory Footprint:** Parameter count and optimizer state memory (Adam stores first and second moments, $3\times$ model weights) grow from `38.4K` parameters (`d_model=16`) to `421.5K` parameters (`d_model=128`, an `11x` increase). Activation memory during backpropagation also scales linearly with $d_{\text{model}}$.
3. **Expressivity vs. Overfitting / Sample Efficiency:**
   - **Too small (`d_model = 16`)**: High bias—fast and lightweight, but suffers from representation bottlenecks and slow optimization, hurting both training (`90.1%`) and test (`79.5%`) accuracy.
   - **Larger (`d_model = 64 -> 128`)**: Higher expressivity and faster convergence improve test accuracy here (`85.2% -> 94.3%` via benign overparameterization and richer features). However, as model size continues to scale without proportional data growth or regularization (dropout, weight decay, early stopping), larger models quickly saturate `100%` training accuracy (`loss = 0.0005`), risking memorization of label noise and poor calibration on small datasets.
