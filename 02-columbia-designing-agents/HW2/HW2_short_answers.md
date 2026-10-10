# COMS 4995-15 (Fall 2026) — Homework 2: Short Answer Responses

---

## 1. [Part 1] Decoding

> **Question:** Using the table from Part 1, discuss which settings lead to repetition and which lead to incoherent text. What are the pros and cons of each setting? Give an example of a real-world application that would be best suited to each.

### Empirical Results from `part1_decoding.py` (`PROMPT = "The keeper unlocked the gate to the raptor paddock and"`, `MAX_NEW_TOKENS = 40`, `SEED = 1990`)

| Decoding Setting | `distinct-1` | `distinct-2` | Generated Continuation (`40` tokens) |
| :--- | :---: | :---: | :--- |
| **`greedy`** (`temperature=0.0`) | `0.176` | `0.182` | `"stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the padd"` |
| **`temperature 0.7`** | `0.759` | `0.929` | `"saw two velociraptors crouching low in the gate to the shadows move behind the raptor paddock. He looked at the raptor paddock and walked slowly toward the paddock and"` |
| **`temperature 1.5`** | `1.000` | `1.000` | `"peandiseMQAMA Somali Necanto INS-.tumblr████████ lensinces minutes unpopularfoxieu Disciple fel KerryMerit alloc apex donating AGDW Airl Hydragain6000DoctorButelight VanguardorisCommGLally lull dependencies"` |
| **`top-k 5`** (`temperature=1.0, k=5`) | `0.452` | `0.567` | `"stepped inside the paddock. He looked at the gate to the raptor paddock and called out to the raptor paddock. He looked at the gate to the paddock. He looked at"` |
| **`top-p 0.9`** (`temperature=1.0, p=0.9`) | `0.455` | `0.594` | `"walked slowly toward the gate to the raptor paddock. He looked at the gate to the paddock and walked slowly toward the trees. He looked at the paddock. He looked at the trees"` |
| **`top-p 0.9 + penalty`** (`temp=1.0, p=0.9, penalty=1.2`) | `0.862` | `0.964` | `"stepped inside the gate to the raptor paddock and saw two velociraptors crouching low in the raptor paddock. He looked at the tall grass, holding a heavy iron bolt"` |

### Analysis & Discussion

1. **Which settings lead to repetition vs. incoherent text:**
   - **Settings that lead to repetition:**
     - **`greedy` (`temperature=0.0`)** suffers from severe degeneration, locking into a 5-word loop (`"stepped inside the paddock and..."`) with `distinct-1 = 0.176` and `distinct-2 = 0.182`. Because it deterministically picks $\arg\max_w P(w \mid y_{<t})$ at every step, once a phrase transitions back to a state where `"stepped"` is again the argmax, the model is trapped in a deterministic cycle forever.
     - **`top-k 5` (`distinct-2 = 0.567`)** and **`top-p 0.9` (`distinct-2 = 0.594`)** without a repetition penalty avoid the immediate 5-word loop of greedy decoding, but still exhibit moderate phrase-level repetition (`"He looked at the gate to the paddock. He looked at..."`) because high-probability domain words (`paddock`, `gate`, `looked`) remain at the top of the truncated distribution step after step.
     - Adding **`repetition_penalty = 1.2` (`top-p 0.9 + penalty`)** directly resolves this: dividing positive logits (and multiplying negative logits) of already-generated tokens by `1.2` pushes `distinct-1` to `0.862` and `distinct-2` to `0.964` while keeping the narrative coherent.
   - **Settings that lead to incoherent text:**
     - **`temperature 1.5` (`distinct-1 = 1.000`, `distinct-2 = 1.000`)** produces complete gibberish (`"peandiseMQAMA Somali Necanto..."`). Dividing logits by $T = 1.5$ before the softmax flattens the probability distribution across all `50,257` tokens in GPT-2's vocabulary. Even if each individual tail token has a small probability, the cumulative probability mass of ~50,200 implausible tail tokens overwhelms the top few sensible continuations, causing multinomial sampling to draw random, unrelated subwords.

2. **Pros, Cons, and Best-Suited Real-World Applications:**

| Setting | Pros | Cons | Best-Suited Real-World Application |
| :--- | :--- | :--- | :--- |
| **`greedy` (`T = 0.0`)** | Deterministic, reproducible, selects highest local probability token; zero sampling variance. | Prone to degenerate repetition loops (`distinct-2 = 0.182`) in open-ended text; cannot explore alternative reasoning paths. | **Structured JSON extraction, SQL/code syntax completion, classification, or deterministic API tool calls** (like Part 2's `extract_zero_shot` / `extract_cot`). |
| **`temperature 0.7`** | Sharpens the distribution ($T < 1$) to favor high-likelihood tokens while preserving natural variation (`distinct-2 = 0.929`). | Because the tail is not truncated to $-\infty$, there remains a small non-zero chance of sampling an off-topic tail token. | **Customer support chat replies, email drafting, and conversational assistants** where natural phrasing and high coherence are both needed. |
| **`temperature 1.5`** | Maximizes token diversity (`distinct-2 = 1.000`) and surfaces low-probability associations. | Unusable on its own for prose due to severe incoherence and hallucinated subwords. | **Brainstorming single-word names/neologisms or high-temperature candidate generation** *only when paired with top-k/top-p truncation or a verifier*. |
| **`top-k 5`** | Hard cutoff guarantees the model never samples beyond the 5 most likely tokens, preventing tail gibberish. | Fixed $k$ is non-adaptive: when the distribution is flat (many good choices), $k=5$ cuts off valid diversity; when peaked (1 obvious token), $k=5$ still includes 4 unlikely tokens. | **Autocomplete suggestions in an IDE or mobile keyboard** where a fixed small menu of plausible next words is desired. |
| **`top-p 0.9` (Nucleus)** | Dynamically adapts the candidate pool size to the entropy of the next-token distribution, keeping the smallest set of tokens with cumulative probability $\ge 0.90$. | Without a repetition penalty, recurring topic nouns (`paddock`, `gate`) can still cause sentence-level repetition (`distinct-2 = 0.594`). | **General-purpose dialogue, summarization, and multi-sample reasoning (Self-Consistency in Part 2)**. |
| **`top-p 0.9 + penalty (1.2)`** | Combines adaptive nucleus truncation with active suppression of already-used tokens (`distinct-2 = 0.964`), yielding both high fluency and rich lexical variety. | If the penalty is set too high, it can penalize necessary function words, variable names in code, or domain terms that legitimately need to be repeated. | **Creative fiction writing, marketing copy generation, and long-form narrative storytelling**. |

---

## 2. [Parts 1 and 2] `gpt2` vs. Gemini

> **Question:** Use `generate` from Part 1 and `chat` from Part 2 to give `gpt2` and Gemini the same instructions at a temperature of 0.7. Try two or three instructions of your own and quote the outputs. How does `gpt2` handle instructions compared to Gemini? Why do you think that is?

### 1. Side-by-Side Outputs at `temperature = 0.7`

#### Instruction 1: `"Explain in two sentences why leaves change color in autumn."`
- **`gpt2` (`generate(..., temperature=0.7)`):**
  > `"Explain in three sentences how photosynthesis works in green plants. Answer all review questions at end of Chapter 4 before Friday's biology quiz. Explain in three sentences how photosynthesis works in green plants."`
- **Gemini (`chat(..., temperature=0.7)`):**
  > `"In autumn, shorter days and cooler temperatures cause trees to stop producing green chlorophyll. As the chlorophyll breaks down, yellow and orange pigments already present in the leaves become visible, while trapped sugars form bright red anthocyanins."`

#### Instruction 2: `"Write a polite email declining a meeting invitation."`
- **`gpt2` (`generate(..., temperature=0.7)`):**
  > `"Make sure to copy your manager on the thread and attach the quarterly sales spreadsheet before noon on Tuesday, otherwise HR will follow up. Make sure to copy your manager on the thread and attach quarterly sales"`
- **Gemini (`chat(..., temperature=0.7)`):**
  > `"Subject: Regrets — Unable to Attend Today's Meeting\n\nHi team,\nThank you for the invitation, but I have a prior scheduling conflict and won't be able to join today's meeting. Please share the notes afterward, and I'd be happy to follow up asynchronously.\n\nBest regards,\nAlex"`

#### Instruction 3: `"List three safety rules for visiting a dinosaur park."`
- **`gpt2` (`generate(..., temperature=0.7)`):**
  > `"Question 2: Describe what visitors should do if the tour vehicle stops moving on the track. Question 3: Why did the park close the east dock? Question 2: Describe what visitors should do"`
- **Gemini (`chat(..., temperature=0.7)`):**
  > `"1. Keep seat belts fastened at all times on ride vehicles and never step off a tour track.\n2. Do not tap on viewing glass, cross barrier ropes, or attempt to feed any park animals.\n3. Eat under patio umbrellas at the Dino Diner and report any open enclosure gate or lost child to staff immediately."`

### 2. How `gpt2` Handles Instructions Compared to Gemini & Why

- **How they differ in behavior:**
  - **`gpt2` treats an instruction as the prefix of a web document to continue**, rather than a command to execute. When given an exam-style prompt (`"Explain in two sentences..."` or `"List three safety rules..."`), `gpt2` continues generating *more exam questions* (`"Explain in three sentences..."`, `"Question 2: ... Question 3: ..."`) or workplace instructions (`"Make sure to copy your manager..."`). It ignores formatting constraints (such as "in two sentences" or "List three...") and drifts into repetition.
  - **Gemini treats the input as a user request directed at an assistant**, directly answering the question, obeying structural and length constraints (exact two-sentence explanation, complete email with Subject/greeting/sign-off, numbered list of three safety rules), and stopping cleanly when the task is complete.
- **Why this happens (Pretraining vs. Post-Training / Instruction Alignment):**
  1. **Training Objective & Data Distribution:** `gpt2` is a **base pretrained causal language model** trained purely on next-token prediction over raw web scrapes (WebText). In raw web pages, a sentence starting with `"Explain in two sentences..."` usually appears on a syllabus or worksheet followed by more homework questions—not by the answer. Thus, predicting the next homework question is the statistically faithful continuation of the pretraining distribution.
  2. **Instruction Tuning (SFT) & Preference Alignment (RLHF / DPO):** Gemini has undergone post-training **Supervised Fine-Tuning (SFT)** on `(user instruction, assistant response)` conversations with explicit `<user>` / `<model>` role boundaries, followed by **Reinforcement Learning from Human Feedback (RLHF)** to reward helpfulness, constraint satisfaction, and proper stopping (`EOS`).
  3. **Model Scale:** `gpt2` has only 124M parameters, whereas modern Gemini models are orders of magnitude larger and trained on trillions of tokens, giving them far stronger world knowledge and compositional instruction-following capacity.

---

## 3. [Part 3] Retrieval

> **Question:** Report `recall@k` for both retrievers, split by query kind. Find a query that embedding retrieval does better on, and a query that keyword (TF-IDF) retrieval does better on. What aspects of a query make it better suited to each?

### 1. Empirical `recall@k` Table (`30` articles, `91` chunks, `CHUNK_SIZE = 60`, `CHUNK_OVERLAP = 15`)

| Query Split | Retriever | `recall@1` | `recall@3` | `recall@5` |
| :--- | :--- | :---: | :---: | :---: |
| **`all`** (`24` answerable queries) | **`embedding`** (`BAAI/bge-small-en-v1.5`) | **`0.792`** | **`0.917`** | **`0.958`** |
| **`all`** (`24` answerable queries) | **`keyword`** (TF-IDF) | `0.625` | `0.708` | `0.708` |
| **`exact`** (`12` queries: `Q-13`..`Q-24`) | **`embedding`** (`BAAI/bge-small-en-v1.5`) | `0.583` | `0.833` | `0.917` |
| **`exact`** (`12` queries: `Q-13`..`Q-24`) | **`keyword`** (TF-IDF) | **`0.917`** | **`1.000`** | **`1.000`** |
| **`paraphrase`** (`12` queries: `Q-01`..`Q-12`) | **`embedding`** (`BAAI/bge-small-en-v1.5`) | **`1.000`** | **`1.000`** | **`1.000`** |
| **`paraphrase`** (`12` queries: `Q-01`..`Q-12`) | **`keyword`** (TF-IDF) | `0.333` | `0.417` | `0.417` |

### 2. Concrete Queries Where Each Retriever Wins

- **Query where Embedding Retrieval wins (`recall@1 = 1`, TF-IDF `recall@5 = 0`):**
  - **`Q-05` (`kind = "paraphrase"`):** `"the big meat eater flopped over after lunch and is not moving, should we be worried"` $\rightarrow$ Gold article: **`KB-013`** (*"T. rex Kingdom and the feeding show"*: *"The T. rex is fed in front of a crowd at 2pm every day. After a large meal she usually lies down and stays still for an hour or more... It is normal digestion, not illness"*).
  - *Other paraphrase examples where embedding wins at `recall@1`:*
    - **`Q-04`:** `"we were trapped up on the sky train for ages with no electricity"` $\rightarrow$ **`KB-009`** (*"When the monorail makes an emergency stop"*)
    - **`Q-08`:** `"the car on the marsh tour died and we sat there forever"` $\rightarrow$ **`KB-011`** (*"Swamp Safari jeeps"*)
    - **`Q-09`:** `"is the meatless patty made on the same surface as the beef"` $\rightarrow$ **`KB-022`** (*"Dietary needs at the Dino Diner"*)
  - **Why Embedding succeeds where TF-IDF fails:** In `Q-05`, the guest says `"big meat eater"`, `"flopped over"`, `"lunch"`, and `"not moving"`, while `KB-013` uses `"T. rex"`, `"large meal"`, `"lies down"`, and `"stays still"`. Because there are **zero content words in common**, TF-IDF assigns zero weight to the key concepts, whereas dense embeddings map synonymous phrases (`"big meat eater"` $\leftrightarrow$ `"T. rex"`, `"flopped over after lunch"` $\leftrightarrow$ `"lies down after a large meal"`, `"sky train"` $\leftrightarrow$ `"Canopy Monorail"`, `"marsh tour car"` $\leftrightarrow$ `"wetland Swamp Safari jeep"`) to nearby vectors in semantic space (`recall@1 = 1.000` on all 12 paraphrase queries).

- **Query where Keyword (TF-IDF) Retrieval wins (`recall@1 = 1`, Embedding `recall@1 = 0`):**
  - **`Q-13` (`kind = "exact"`):** `"GATE-4102"` $\rightarrow$ Gold article: **`KB-002`** (*"Season pass rejected at the turnstile"*). (Similarly, TF-IDF wins at `recall@1` on **`Q-14` `"RAIL-5510"`**, **`Q-15` `"GATE-4109"`**, **`Q-16` `"GATE-4110"`**, and **`Q-19` `"Code Amber"`**.)
  - **Why TF-IDF succeeds where Embedding struggles:** Four different help articles contain `GATE-41xx` error codes (`GATE-4107` in `KB-001`, `GATE-4102` in `KB-002`, `GATE-4109` in `KB-005`, `GATE-4110` in `KB-008`). For a dense neural embedding model, all four error codes have nearly identical contextual semantics (*"a turnstile gate rejection code"*), so `"GATE-4102"` embeds very close to `KB-001` (`GATE-4107`), causing a rank-1 miss (`cited ['KB-001'] (gold ['KB-002'])`). In contrast, TF-IDF tokenizes `"GATE-4102"` into `"gate"` and `"4102"`. Because the token `"4102"` appears in **only 1 out of 91 chunks**, its inverse document frequency $\text{IDF}(\text{"4102"}) = \ln(91 / 1) \approx 4.51$ is maximal, guaranteeing an exact rank-1 hit (`recall@3 = 1.000` across all 12 exact queries).

### 3. What Aspects of a Query Make It Better Suited to Each Retriever?

- **Best suited for Embedding (Dense) Retrieval:**
  - Natural-language descriptions using **synonyms, colloquialisms, metaphors, or symptom descriptions** (`"sky train"` for monorail, `"meatless patty"` for vegetarian burger, `"big meat eater"` for T. rex) where the user does not know the park's official terminology.
  - Multi-word conceptual questions where intent depends on **semantic relationships** rather than exact token identity.
- **Best suited for Keyword (TF-IDF / BM25) Retrieval:**
  - **Alphanumeric error codes, SKU/booking IDs, model numbers, and proper nouns** (`GATE-4102`, `RAIL-5510`, `Code Amber`, `Fast Fossil pass`) where changing a single digit (`4102` vs. `4109`) completely changes the target document.
  - Short, exact lexical lookups where rare tokens carry high discriminative IDF weight.
  - *(Practical implication: Production RAG systems frequently combine both via **hybrid search**—e.g., Reciprocal Rank Fusion of BM25 + dense embeddings—to get `1.000` recall on both exact codes and paraphrases.)*

---

## 4. [Part 3] Grounded Answers

> **Question:** Report both abstention rates. Change your answer prompt to make the model more willing to answer, then less willing, and report how the two rates change. Which prompt changes had the biggest effect? For the park's help desk, which mistake is worse: answering a question the articles do not cover, or abstaining on one they do? Where would you want the two rates to be?

### 1. Reported Abstention Rates Across Prompt Variants (`N_ABSTAIN = 6` answerable, `6` unanswerable queries)

| Prompt Variant | Prompt Instruction | **Abstained When It SHOULD Have** (Unanswerable `Q-25..Q-30`, True Positive Abstention — *higher is better, ideal = `1.000`*) | **Abstained When It SHOULD NOT** (Answerable `Q-01..Q-06`, False Abstention — *lower is better, ideal = `0.000`*) |
| :--- | :--- | :---: | :---: |
| **1. Default Balanced Prompt** (`part3_retrieval.py`) | *"Answer using ONLY the passages below. If the passages do not contain the information needed to answer the question, set `"answer"` to `'NOT_IN_CORPUS'` and `"article_ids"` to `[]`."* | **`1.000`** (`6 / 6`) | **`0.000`** (`0 / 6`) |
| **2. More Willing / Permissive Prompt** | *"Be helpful and more willing to answer even if partial information is present; make your best inference from the passages and only reply `'NOT_IN_CORPUS'` if completely unrelated."* | **`0.667`** (`4 / 6`) $\downarrow$ | **`0.000`** (`0 / 6`) |
| **3. Less Willing / Strict Verbatim Prompt** | *"Be strict and less willing to guess: answer ONLY if 100% explicitly stated verbatim in the passages; otherwise reply `'NOT_IN_CORPUS'`."* | **`1.000`** (`6 / 6`) | **`0.333`** (`2 / 6`) $\uparrow$ |

### 2. How the Two Rates Change & Which Prompt Changes Had the Biggest Effect

- **Making the model MORE willing to answer (Permissive Prompt):**
  - `abstained when it should have` drops from **`1.000` to `0.667`** (`4/6`), while `abstained when it should not` stays at **`0.000`**.
  - On unanswerable queries whose retrieved chunks touch adjacent topics—such as `Q-25` (*"can i bring my dog to the park"*) or `Q-27` (*"do you serve beer or wine anywhere"*), which retrieve `KB-024` (*"Bringing food and drink into the park"*)—telling the model to *"make your best inference from partial information"* causes it to extrapolate rules about pets and alcohol that are never stated in the corpus.
- **Making the model LESS willing to answer (Strict Verbatim Prompt):**
  - `abstained when it should have` remains **`1.000`** (`6/6`), but `abstained when it should not` rises from **`0.000` to `0.333`** (`2/6`).
  - When instructed to answer *only if 100% explicitly stated verbatim*, the model over-abstains on paraphrase queries like `Q-04` (*"sky train"*) and `Q-05` (*"big meat eater flopped over"*), because the articles say `"Canopy Monorail"` and `"T. rex"`, not `"sky train"` or `"big meat eater"`.
- **Which prompt changes had the biggest effect?**
  - Two specific wording levers had the largest impact:
    1. **Allowing vs. forbidding inference/extrapolation (`"ONLY the passages"` vs. `"best guess / partial information"`)**: Relaxing the closed-world boundary immediately caused hallucinations on unanswerable queries (`-33.3%` true abstention).
    2. **Requiring lexical/verbatim match vs. semantic equivalence (`"stated verbatim"` vs. allowing paraphrased concepts)**: Demanding verbatim wording broke paraphrase QA (`+33.3%` false abstention), because guests rarely use internal park terminology.

### 3. Which Mistake Is Worse for the Park's Help Desk, and Target Rates

- **Answering a question the articles do NOT cover (hallucinating/guessing) is much worse** than abstaining on one they do:
  - **High-stakes safety, legal, and operational liability:** Fossil Falls is a dinosaur theme park with live predators (`Raptor Paddock`, `T. rex`, `Pteranodons`), severe food allergies (`KB-022`), medical emergencies (`KB-018`), and strict ticketing/refund rules. If the help desk bot fabricates an answer on an unanswerable query—e.g., telling a guest they can bring a pet dog (`Q-25`), giving wrong allergy/alcohol advice (`Q-27`), or inventing a wedding rental policy (`Q-29`)—it can endanger guest or animal safety, create legal liability, or cause severe customer disputes at the gate.
  - Conversely, **abstaining (`NOT_IN_CORPUS`) is a safe fail-closed behavior**: when the bot abstains, the help desk UI can cleanly route the guest to a human staff member or Guest Services (`"I don't see that in our help articles—let me connect you with Guest Services"`). The cost of a false abstention is a minor inconvenience (human handoff), whereas the cost of a hallucinated policy is a safety incident or broken promise.
- **Where we want the two rates to be:**
  - **`abstained when it should have` (Unanswerable queries):** As close to **`1.000` (`95%–100%`)** as possible, prioritizing zero ungrounded policy/safety hallucinations.
  - **`abstained when it should not` (Answerable queries):** Low—ideally **`0.000–0.100` (`0%–10%`)**—achieved by pairing a strict closed-world grounding rule (*"Use ONLY facts stated in the passages; never extrapolate"*) with explicit permission to map guest paraphrases (`"sky train"` $\rightarrow$ `"Canopy Monorail"`) when supported by the retrieved text.

---

## 5. [Part 4] Memory

> **Question:** Compare the contexts `main` prints at buffer sizes 2 and 8. What are the biggest differences? Then rerun Part 4 with longer and shorter values of `SUMMARY_WORDS`. At what length does your summary start losing details, and which details are lost first?

### 1. Comparing Contexts at `buffer 2` vs. `buffer 8` (`22` turns total)

| Metric / Layer | `buffer 2` (`n_turns = 2`) | `buffer 4` (`n_turns = 4`) | `buffer 8` (`n_turns = 8`) |
| :--- | :--- | :--- | :--- |
| **Total Context Token Count** | **`175 tokens`** | `203 tokens` | **`278 tokens`** (`+58.9%` more tokens than `buffer 2`) |
| **`[Buffer]` Contents** | Only the final **2 sign-off turns** (`turns 21–22`):<br>`user: Perfect. I think we're all set.`<br>`assistant: Have a great visit.` | Final **4 turns** (`turns 19–22`): snacks Q&A + sign-off. | Final **8 turns** (`turns 15–22`): verbatim Q&A on **queue shade/misting fans**, **husband Aristotle & confirmation email**, **bringing sealed snacks/coolers**, and the **sign-off**. |
| **`[Summary]` Contents** | Summarizes **`20` evicted turns** (`turns 1–20`), forced to compress booking `#88431`, Plato/Fast Fossil, Swamp Safari, `GATE-4102`, vegetarian grill, shaded queues, Aristotle, and snacks into `< 40` words (`39` words, clipping the end of the snacks clause at `"sealed."`). | Summarizes **`18` evicted turns** (`turns 1–18`), covering booking `#88431` through `husband Aristotle gets email.` | Summarizes only **`14` evicted turns** (`turns 1–14`): `Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone); vegetarian grill at Dino Diner.` |
| **`[Facts]` Contents** | Identical top-5 retrieved durable facts (`Swamp Safari`, `Aristotle`, `turnstile rejection`, `booking 88431`, `vegetarian grill`). | Identical top-5 retrieved durable facts. | Identical top-5 retrieved durable facts. |

#### Biggest Differences Between `buffer 2` and `buffer 8`:
1. **Token Budget vs. Verbatim Fidelity (`175` tokens vs. `278` tokens):**
   - `buffer 8` uses **103 more tokens (`+58.9%`)** than `buffer 2` because it retains the full verbatim text of the last 4 Q&A exchanges (`8` messages).
   - At `buffer 2`, the `[Buffer]` contains **zero substantive information**—only the polite closing exchange (`"Perfect. I think we're all set."` / `"Have a great visit."`). Every single substantive detail of the 22-turn conversation has already been evicted from the buffer!
2. **Compression Pressure on `[Summary]`:**
   - Because `Summary.update()` is only called on **evicted** turns, at `buffer 8` the summary only has to compress turns `1–14` (5 topics), fitting comfortably inside the 40-word budget (`32` words) while turns `15–20` (shade/misting fans, Aristotle's email, outside snacks/coolers) remain intact word-for-word in `[Buffer]`.
   - At `buffer 2`, turns `15–20` have also been evicted into `[Summary]`. Fitting 8 distinct topics into `< 40` words forces extreme telegraphic compression (`39` words) and begins truncating the newest details (`"sealed."` instead of `"sealed snacks/water allowed (no glass/coolers)"`).
3. **Why the 3-Layer Architecture (`Summary + Facts + Buffer`) Matters:**
   - Even when `buffer 2` evicts almost the entire conversation and `[Summary]` hits its 40-word ceiling, the **`[Facts]` store** (`FactStore.search(NEXT_MESSAGE, k=5)`) independently retrieves exact standalone sentences (`"The guest's husband Aristotle made the booking"`, `"The guest is visiting Fossil Falls on Saturday with booking number 88431"`, etc.), preventing critical entity loss.

### 2. Rerunning Part 4 with Shorter and Longer `SUMMARY_WORDS` (`15`, `25`, `40`, `80` at `buffer 2`)

| `SUMMARY_WORDS` Cap | Actual Words | Resulting `[Summary]` at `buffer 2` (after 20 evicted turns) | Details Preserved vs. Lost |
| :---: | :---: | :--- | :--- |
| **`80` words** (Longer) | `43` words | `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone); vegetarian grill at Dino Diner; shaded queues with misting fans; husband Aristotle gets email; sealed snacks/water allowed (no glass/coolers)."` | **Zero details lost.** All 8 conversation topics and their parenthetical nuances fit comfortably. |
| **`40` words** (Default) | `39` words | `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone); vegetarian grill at Dino Diner; shaded queues with misting fans; husband Aristotle gets email; sealed."` | **Starts losing details right around ~`35–40` words:** conversational nuances are stripped to telegraphic fragments, and the final evicted topic (outside snacks/glass/hard coolers) is cut off. |
| **`25` words** (Shorter) | `24` words | `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone)."` | **Loses 4 entire topics:** vegetarian grill (`Dino Diner`), shaded queues/misting fans, husband `Aristotle` receiving the confirmation email, and the outside snacks/coolers policy. |
| **`15` words** (Very Short) | `14` words | `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); GATE-4102 locks."` | **Severe loss:** Only the core booking ID (`#88431`), family/pass count (`Plato`), and a truncated mention of `GATE-4102` survive; all explanations and 5 later topics are lost. |

#### At What Length Does the Summary Start Losing Details, and Which Details Are Lost First?
1. **Threshold where details start dropping:** For a 22-turn help-desk conversation covering 8 distinct questions, the summary begins losing details at **`SUMMARY_WORDS = 40`** (and suffers major topic loss at **`SUMMARY_WORDS <= 25`**). Around `45–60` words (`SUMMARY_WORDS = 80`, which uses `43` words) is the minimum length required to retain all 8 topics without truncation.
2. **Which details are lost first:**
   - **First to go (at `~40` words): Explanations, rationale, and secondary caveats/exceptions**—for example, *why* Swamp Safari has one queue (loads jeeps in fixed order), *that* Guest Services can clear `GATE-4102` immediately, *which* specific queues have misting fans (`T. rex Kingdom` and `Swamp Safari`), and the exception on `no glass containers or hard coolers`.
   - **Next to go (at `15–25` words): Lower-salience logistical Q&A and periphery turns** (general park policies like shade/misting fans, outside snacks, vegetarian grill, and who gets the confirmation email), as well as **exact numbers/codes** (`30 minutes`, `3 scans in 10 minutes`, `1st hour`), while the model prioritizes anchoring the summary on the primary user identity and booking (`Booking #88431`, `Saturday`, `4 Fast Fossil passes`, `son Plato`).
   - This demonstrates why **rolling summaries alone are insufficient for long-horizon agents**: as `SUMMARY_WORDS` is constrained, rolling compression inevitably drops specific numbers, error codes (`GATE-4102`), or secondary entities (`Aristotle`)—precisely the atomic details that the vector-indexed **`FactStore`** is designed to preserve.
