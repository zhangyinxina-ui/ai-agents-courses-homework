1. **[Part 1] Decoding.** Using the table from Part 1, discuss which settings lead to repetition and which lead to incoherent text. What are the pros and cons of each setting? Give an example of a real-world application that would be best suited to each.

**Answer:**

**Empirical Table from `part1_decoding.py` (`PROMPT = "The keeper unlocked the gate to the raptor paddock and"`, `MAX_NEW_TOKENS = 40`, `SEED = 1990`):**
- **`greedy` (`temperature=0.0`)**: `distinct-1 = 0.176`, `distinct-2 = 0.182`
  - Output: `"stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the paddock and stepped inside the padd"`
- **`temperature 0.7`**: `distinct-1 = 0.759`, `distinct-2 = 0.929`
  - Output: `"saw two velociraptors crouching low in the gate to the shadows move behind the raptor paddock. He looked at the raptor paddock and walked slowly toward the paddock and"`
- **`temperature 1.5`**: `distinct-1 = 1.000`, `distinct-2 = 1.000`
  - Output: `"peandiseMQAMA Somali Necanto INS-.tumblr████████ lensinces minutes unpopularfoxieu Disciple fel KerryMerit alloc apex donating AGDW Airl Hydragain6000DoctorButelight VanguardorisCommGLally lull dependencies"`
- **`top-k 5` (`temperature=1.0, k=5`)**: `distinct-1 = 0.452`, `distinct-2 = 0.567`
  - Output: `"stepped inside the paddock. He looked at the gate to the raptor paddock and called out to the raptor paddock. He looked at the gate to the paddock. He looked at"`
- **`top-p 0.9` (`temperature=1.0, p=0.9`)**: `distinct-1 = 0.455`, `distinct-2 = 0.594`
  - Output: `"walked slowly toward the gate to the raptor paddock. He looked at the gate to the paddock and walked slowly toward the trees. He looked at the paddock. He looked at the trees"`
- **`top-p 0.9 + penalty` (`temperature=1.0, p=0.9, penalty=1.2`)**: `distinct-1 = 0.862`, `distinct-2 = 0.964`
  - Output: `"stepped inside the gate to the raptor paddock and saw two velociraptors crouching low in the raptor paddock. He looked at the tall grass, holding a heavy iron bolt"`

**Discussion:**
1. **Repetition vs. Incoherence:**
   - **Repetition:** `greedy` (`distinct-2 = 0.182`) falls into a deterministic 5-word loop (`"stepped inside the paddock and..."`) because picking the argmax token at every step traps the model in a cycle once a state repeats. Unpenalized `top-k 5` (`0.567`) and `top-p 0.9` (`0.594`) avoid immediate token loops but still repeat high-probability domain phrases (`"He looked at the paddock"`). Adding `repetition_penalty = 1.2` (`top-p 0.9 + penalty`) suppresses already-generated token logits and raises `distinct-2` to `0.964` while preserving narrative coherence.
   - **Incoherence:** `temperature 1.5` (`distinct-2 = 1.000`) produces incoherent subword salad (`"peandiseMQAMA Somali Necanto..."`) because dividing logits by `1.5` flattens the softmax distribution across all `50,257` vocabulary tokens, giving the ~50,200 implausible tail tokens enough combined probability mass to dominate sampling.
2. **Pros, Cons, and Real-World Applications:**
   - **`greedy` (`T=0.0`)**: *Pros:* Deterministic, reproducible, zero variance. *Cons:* Degenerate repetition in open-ended generation. *Best application:* **Structured JSON extraction (`Part 2`), SQL/code generation, or deterministic tool calling**.
   - **`temperature 0.7`**: *Pros:* Concentrates probability mass on high-likelihood tokens while allowing natural phrasing (`distinct-2 = 0.929`). *Cons:* Non-zero tail probability without top-k/top-p truncation. *Best application:* **Customer service chatbots and business email drafting**.
   - **`temperature 1.5`**: *Pros:* Maximum lexical diversity (`distinct-2 = 1.000`). *Cons:* Severe incoherence if un-truncated. *Best application:* **Creative brainstorming of single-word names/ideas when paired with a filter/verifier**.
   - **`top-k 5`**: *Pros:* Hard cutoff prevents tail gibberish. *Cons:* Fixed $k=5$ cannot adapt when the distribution is very peaked (1 valid token) or very flat (20 valid tokens). *Best application:* **Keyboard/IDE autocomplete menus**.
   - **`top-p 0.9` (Nucleus)**: *Pros:* Dynamically scales candidate pool size with distribution entropy. *Cons:* Can still repeat common topic words over long passages (`distinct-2 = 0.594`). *Best application:* **General conversational assistants, summarization, and Self-Consistency sampling (`Part 2`)**.
   - **`top-p 0.9 + penalty (1.2)`**: *Pros:* High fluency plus strong lexical diversity (`distinct-2 = 0.964`). *Cons:* Can over-penalize terms that legitimately need repeating (e.g., variable names in code). *Best application:* **Creative story generation and long-form marketing copy**.

---

2. **[Parts 1 and 2] gpt2 vs. Gemini.** Use generate from Part 1 and chat from Part 2 to give gpt2 and Gemini the same instructions at a temperature of 0.7. Try two or three instructions of your own and quote the outputs. How does gpt2 handle instructions compared to Gemini? Why do you think that is?

**Answer:**

**Quoted Outputs at `temperature = 0.7`:**
1. **Instruction 1:** `"Explain in two sentences why leaves change color in autumn."`
   - **`gpt2` (`generate`)**: `"Explain in three sentences how photosynthesis works in green plants. Answer all review questions at end of Chapter 4 before Friday's biology quiz. Explain in three sentences how photosynthesis works in green plants."`
   - **Gemini (`chat`)**: `"In autumn, shorter days and cooler temperatures cause trees to stop producing green chlorophyll. As the chlorophyll breaks down, yellow and orange pigments already present in the leaves become visible, while trapped sugars form bright red anthocyanins."`
2. **Instruction 2:** `"Write a polite email declining a meeting invitation."`
   - **`gpt2` (`generate`)**: `"Make sure to copy your manager on the thread and attach the quarterly sales spreadsheet before noon on Tuesday, otherwise HR will follow up. Make sure to copy your manager on the thread and attach quarterly sales"`
   - **Gemini (`chat`)**: `"Subject: Regrets — Unable to Attend Today's Meeting\n\nHi team,\nThank you for the invitation, but I have a prior scheduling conflict and won't be able to join today's meeting. Please share the notes afterward, and I'd be happy to follow up asynchronously.\n\nBest regards,\nAlex"`
3. **Instruction 3:** `"List three safety rules for visiting a dinosaur park."`
   - **`gpt2` (`generate`)**: `"Question 2: Describe what visitors should do if the tour vehicle stops moving on the track. Question 3: Why did the park close the east dock? Question 2: Describe what visitors should do"`
   - **Gemini (`chat`)**: `"1. Keep seat belts fastened at all times on ride vehicles and never step off a tour track.\n2. Do not tap on viewing glass, cross barrier ropes, or attempt to feed any park animals.\n3. Eat under patio umbrellas at the Dino Diner and report any open enclosure gate or lost child to staff immediately."`

**Comparison & Explanation:**
- **How `gpt2` handles instructions vs. Gemini:** `gpt2` does not execute the instruction; instead, it **continues the prompt as a document** (e.g., generating the next questions on a biology quiz or worksheet, or adding another directive to an office memo) and ignores length/format constraints. Gemini recognizes the prompt as a user request, directly fulfills the task, respects exact constraints ("two sentences", "polite email", "three safety rules"), and stops cleanly.
- **Why:** `gpt2` (124M parameters) is a **base pretrained language model** trained solely with a causal next-token prediction objective on raw web text (WebText), where imperative sentences like `"Explain in two sentences..."` usually appear in lists of exam questions rather than immediately before an assistant's answer. Gemini has undergone **Supervised Instruction Fine-Tuning (SFT)** on multi-turn `(user, model)` chat transcripts and **Reinforcement Learning from Human Feedback (RLHF)** to align the model to act as a helpful, instruction-following assistant.

---

3. **[Part 3] Retrieval.** Report recall@k for both retrievers, split by query kind. Find a query that embedding retrieval does better on, and a query that keyword (TF-IDF) retrieval does better on. What aspects of a query make it better suited to each?

**Answer:**

**1. Reported `recall@k` by Query Kind (`30` articles, `91` chunks):**

| Split | Retriever | `recall@1` | `recall@3` | `recall@5` |
| :--- | :--- | :---: | :---: | :---: |
| **`all`** (`24` answerable) | `embedding` (`BAAI/bge-small-en-v1.5`) | **`0.792`** | **`0.917`** | **`0.958`** |
| **`all`** (`24` answerable) | `keyword` (TF-IDF) | `0.625` | `0.708` | `0.708` |
| **`exact`** (`12` queries) | `embedding` (`BAAI/bge-small-en-v1.5`) | `0.583` | `0.833` | `0.917` |
| **`exact`** (`12` queries) | `keyword` (TF-IDF) | **`0.917`** | **`1.000`** | **`1.000`** |
| **`paraphrase`** (`12` queries) | `embedding` (`BAAI/bge-small-en-v1.5`) | **`1.000`** | **`1.000`** | **`1.000`** |
| **`paraphrase`** (`12` queries) | `keyword` (TF-IDF) | `0.333` | `0.417` | `0.417` |

**2. Specific Query Examples:**
- **Where Embedding Retrieval excels (`recall@1 = 1`, Keyword fails):**
  - **`Q-05` (`paraphrase`):** `"the big meat eater flopped over after lunch and is not moving, should we be worried"` (Gold: **`KB-013`**, *"T. rex Kingdom and the feeding show"*). Also `Q-04` (`"trapped up on the sky train... no electricity"` $\rightarrow$ `KB-009` monorail), `Q-08` (`"car on the marsh tour died"` $\rightarrow$ `KB-011` Swamp Safari jeeps), and `Q-09` (`"meatless patty made on the same surface as the beef"` $\rightarrow$ `KB-022` vegetarian grill).
  - *Why:* `Q-05` shares **zero content words** with `KB-013` (`"big meat eater"` vs. `"T. rex"`, `"flopped over after lunch"` vs. `"lies down after a large meal"`). TF-IDF gives zero score to non-overlapping synonyms, whereas dense embeddings map semantically equivalent concepts to nearby vectors (`1.000` recall on all 12 paraphrases).
- **Where Keyword (TF-IDF) Retrieval excels (`recall@1 = 1`, Embedding misses at `k=1`):**
  - **`Q-13` (`exact`):** `"GATE-4102"` (Gold: **`KB-002`**, *"Season pass rejected at the turnstile"*), as well as `Q-14` (`"RAIL-5510"`), `Q-15` (`"GATE-4109"`), `Q-16` (`"GATE-4110"`), and `Q-19` (`"Code Amber"`).
  - *Why:* Four different articles contain `GATE-41xx` error codes (`4107`, `4102`, `4109`, `4110`). Dense embeddings cluster all four codes together as *"gate error code"* and rank `KB-001` (`GATE-4107`) first on `Q-13`. TF-IDF splits `"GATE-4102"` into `"gate"` and `"4102"`, where `"4102"` appears in only 1 chunk out of 91 ($\text{IDF} = \ln(91/1) \approx 4.51$), pinpointing `KB-002` at rank 1.

**3. Query Aspects Better Suited to Each:**
- **Embedding retrieval** is best for **natural-language paraphrases, colloquial descriptions, and conceptual queries** where users describe symptoms or meanings in their own words without knowing official terminology.
- **Keyword (TF-IDF) retrieval** is best for **exact error codes (`GATE-4102`), IDs, product names (`Code Amber`), and rare technical identifiers** where a single character or digit difference changes the meaning entirely.

---

4. **[Part 3] Grounded answers.** Report both abstention rates. Change your answer prompt to make the model more willing to answer, then less willing, and report how the two rates change. Which prompt changes had the biggest effect? For the park's help desk, which mistake is worse: answering a question the articles do not cover, or abstaining on one they do? Where would you want the two rates to be?

**Answer:**

**1. Abstention Rates Across Prompt Variants (`6` answerable, `6` unanswerable queries):**
- **Default Prompt** (*"Answer using ONLY the passages below. If they do not answer it, set `"answer"` to `'NOT_IN_CORPUS'`"*):
  - `abstained when it should have` (unanswerable): **`1.000`** (`6/6`)
  - `abstained when it should not` (answerable): **`0.000`** (`0/6`)
- **More Willing / Permissive Prompt** (*"Be helpful and more willing to answer even if partial information is present; make your best guess from the passages..."*):
  - `abstained when it should have` (unanswerable): **`0.667`** (`4/6`) — drops by `33.3%` as the model extrapolates unstated rules on `Q-25` (dogs) and `Q-27` (beer/wine) from `KB-024` (bringing food/drink).
  - `abstained when it should not` (answerable): **`0.000`** (`0/6`)
- **Less Willing / Strict Verbatim Prompt** (*"Be strict and less willing to guess: answer ONLY if 100% explicitly stated verbatim in the passages..."*):
  - `abstained when it should have` (unanswerable): **`1.000`** (`6/6`)
  - `abstained when it should not` (answerable): **`0.333`** (`2/6`) — rises by `33.3%` because the model refuses to answer slang/paraphrase queries (`"sky train"`, `"big meat eater"`) that do not appear verbatim in the articles.

**2. Which Prompt Changes Had the Biggest Effect:**
- (a) **Permitting inference from partial context (`"best guess / partial information"` vs. `"ONLY the passages"`)** had the biggest effect on unanswerable queries, immediately causing policy hallucinations when retrieved chunks were topically adjacent (`KB-024` food rules vs. pets/alcohol).
- (b) **Requiring verbatim wording (`"100% explicitly stated verbatim"`)** had the biggest effect on answerable queries, causing false abstentions whenever guests used everyday paraphrases instead of official park terms.

**3. Which Mistake Is Worse for the Park's Help Desk & Target Rates:**
- **Answering a question the articles do NOT cover (hallucination) is far worse** than abstaining on one they do. Fossil Falls deals with physical safety around live dinosaurs (`Raptor Paddock`, `Pteranodons`), severe food allergies (`KB-022`), lost children (`Code Amber`), and binding refund/pass rules. Hallucinating a rule (e.g., claiming dogs are allowed, guaranteeing nut-free buns, or inventing ride safety procedures) creates severe safety and liability risks. Abstaining (`NOT_IN_CORPUS`) fails safely by escalating the guest to a human agent at Guest Services.
- **Target Rates:** We want `abstained when it should have` (true abstention on unanswerable queries) at **`0.95–1.00` (`~100%`)**, and `abstained when it should not` (false abstention on answerable queries) low at **`0.00–0.10` (`0%–10%`)**.

---

5. **[Part 4] Memory.** Compare the contexts main prints at buffer sizes 2 and 8. What are the biggest differences? Then rerun Part 4 with longer and shorter values of SUMMARY_WORDS. At what length does your summary start losing details, and which details are lost first?

**Answer:**

**1. Comparison of Contexts at Buffer Size `2` vs. Buffer Size `8` (`22` turns total):**
- **Token Count & Buffer Content (`175` tokens at `buffer 2` vs. `278` tokens at `buffer 8`):**
  - At **`buffer 2` (`175` tokens)**, `[Buffer]` holds only the final 2 polite sign-off turns (`"Perfect. I think we're all set."` / `"Have a great visit."`), containing **zero substantive facts**. All 20 prior turns have been evicted into `[Summary]`.
  - At **`buffer 8` (`278` tokens, `+58.9%` larger)**, `[Buffer]` retains the last 8 turns verbatim, preserving the exact Q&A dialogue about **covered queues with misting fans (`T. rex Kingdom` and `Swamp Safari`)**, **husband `Aristotle` receiving the confirmation email**, and **bringing sealed snacks/water bottles (no glass or hard coolers)**.
- **Summary Compression Load:**
  - At **`buffer 8`**, `[Summary]` only compresses the first 14 evicted turns (`Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone); vegetarian grill at Dino Diner.`), fitting cleanly in `32` words.
  - At **`buffer 2`**, `[Summary]` must compress 20 evicted turns (all 8 topics) under the 40-word cap (`39` words), forcing telegraphic phrasing and clipping the end of the snacks policy (`"...husband Aristotle gets email; sealed."`).
- **Role of `[Facts]`:** In both settings, `[Facts]` retrieves the same top-5 standalone facts for `NEXT_MESSAGE`, acting as a precision safety net for specific entities (`#88431`, `Aristotle`, `Swamp Safari`, `vegetarian grill`) that either scrolled out of `[Buffer]` or got compressed in `[Summary]`.

**2. Rerunning Part 4 Across `SUMMARY_WORDS` (`15`, `25`, `40`, `80` at `buffer 2`):**
- **`SUMMARY_WORDS = 80` (`43` words used):**
  > `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone); vegetarian grill at Dino Diner; shaded queues with misting fans; husband Aristotle gets email; sealed snacks/water allowed (no glass/coolers)."`
  - Preserves all 8 topics and parenthetical caveats without loss.
- **`SUMMARY_WORDS = 40` (`39` words used — Default):**
  > `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone); vegetarian grill at Dino Diner; shaded queues with misting fans; husband Aristotle gets email; sealed."`
  - **Starts losing details at `~40` words**: full sentences become fragments and the final topic (`snacks/glass/coolers`) is truncated.
- **`SUMMARY_WORDS = 25` (`24` words used):**
  > `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); Swamp Safari 1st hour; GATE-4102 locks 30m after 3 scans (1 pass/phone)."`
  - **Loses 4 entire topics:** vegetarian grill, queue shade/misting fans, husband Aristotle's confirmation email, and outside snacks/coolers.
- **`SUMMARY_WORDS = 15` (`14` words used):**
  > `"Booking #88431 Saturday; 4 Fast Fossil passes (son Plato, 8, loves raptors); GATE-4102 locks."`
  - Retains only the primary booking anchor (`#88431`, `Saturday`, `4 Fast Fossil passes`, `Plato`) and a stub of `GATE-4102`.

**3. When Details Are Lost & Which Disappear First:**
- The summary begins losing details around **`SUMMARY_WORDS = 40`** (with severe topic loss at **`<= 25` words**); at least `~45–50` words are needed to summarize all 20 evicted turns intact.
- **Lost first (at `40` words):** **Explanations, qualifiers, and negative exceptions** (e.g., *why* Swamp Safari is excluded, *that* Guest Services clears `GATE-4102`, *which* specific queues have misting fans, and the ban on *glass/hard coolers*).
- **Lost next (at `15–25` words):** **General park policy Q&A and secondary/later turns** (snacks, shade, vegetarian grill, confirmation email recipient `Aristotle`) and **exact numerical parameters** (`30m`, `3 scans in 10m`), while the core booking identity (`Booking #88431`, `Saturday`, `4 passes`, `son Plato`) is preserved longest.
