"""Run all HW2 experiments (Parts 1-4 + Q2, Q4, Q5 ablation studies) and save results to hw2_experiment_results.json."""

from __future__ import annotations

import json
from pathlib import Path

import part1_decoding as p1
import part2_prompting as p2
import part3_retrieval as p3
import part4_memory as p4

HERE = Path(__file__).resolve().parent


def main() -> None:
    # Clear cache so updated summary logic is used and re-cached cleanly
    if p2.CACHE.exists():
        p2.CACHE.unlink()

    results: dict = {}

    # --- 1. Part 1 Decoding ---
    model, tokenizer = p1.load_model()
    logits = p1.next_token_logits(model, tokenizer, p1.PROMPT)
    probs = p1.apply_temperature(logits, 1.0)
    top_p, top_i = probs.topk(6)
    top_tokens = [
        {"token": tokenizer.decode([int(i)]), "prob": round(float(p), 4)}
        for p, i in zip(top_p, top_i)
    ]

    decoding_table = []
    for name, kwargs in p1.SETTINGS:
        text = p1.generate(model, tokenizer, p1.PROMPT, p1.MAX_NEW_TOKENS, **kwargs)
        continuation = text[len(p1.PROMPT) :].strip().replace("\n", " ")
        d1 = round(p1.distinct_n(continuation, 1), 3)
        d2 = round(p1.distinct_n(continuation, 2), 3)
        decoding_table.append(
            {
                "setting": name,
                "kwargs": kwargs,
                "distinct_1": d1,
                "distinct_2": d2,
                "continuation": continuation,
            }
        )
    results["part1"] = {"top_tokens": top_tokens, "settings": decoding_table}

    # --- 2. Q2: gpt2 vs Gemini at temperature=0.7 ---
    instructions = [
        "Explain in two sentences why leaves change color in autumn.",
        "Write a polite email declining a meeting invitation.",
        "List three safety rules for visiting a dinosaur park.",
    ]
    q2_results = []
    for inst in instructions:
        gpt2_full = p1.generate(model, tokenizer, inst, max_new_tokens=40, temperature=0.7)
        gpt2_reply = gpt2_full[len(inst) :].strip().replace("\n", " ")
        gemini_reply = p2.chat([{"role": "user", "content": inst}], temperature=0.7)
        q2_results.append(
            {
                "instruction": inst,
                "gpt2_temp_0_7": gpt2_reply,
                "gemini_temp_0_7": gemini_reply,
            }
        )
    results["q2_gpt2_vs_gemini"] = q2_results

    # --- 3. Part 2 Structured Prompting ---
    tickets, gold = p2.load_tickets()
    tickets = tickets[: p2.N_SCORE]
    p2_summary = {}
    for name, fn in [
        ("zero-shot", p2.extract_zero_shot),
        ("few-shot", p2.extract_few_shot),
        ("cot", p2.extract_cot),
        ("consistent", p2.extract_consistent),
    ]:
        preds = {t["id"]: fn(t) for t in tickets}
        p2_summary[name] = {
            "validity_rate": round(p2.validity_rate(preds), 3),
            "field_accuracy": {k: round(v, 3) for k, v in p2.field_accuracy(preds, gold).items()},
        }
    results["part2"] = p2_summary

    # --- 4. Part 3 Retrieval & Q4 Grounded Answers ---
    articles = p3.load_articles()
    queries = [
        json.loads(line)
        for line in p3.QUERIES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks, matrix = p3.build_index(articles)
    answerable = [q for q in queries if q["kind"] != "unanswerable"]
    unanswerable = [q for q in queries if q["kind"] == "unanswerable"]

    recall_table = {}
    for label, subset in [
        ("all", answerable),
        ("exact", [q for q in queries if q["kind"] == "exact"]),
        ("paraphrase", [q for q in queries if q["kind"] == "paraphrase"]),
    ]:
        recall_table[label] = {
            "embedding": {
                f"recall@{k}": round(p3.recall_at_k(subset, chunks, matrix, k, "embedding"), 3)
                for k in p3.K_VALUES
            },
            "keyword": {
                f"recall@{k}": round(p3.recall_at_k(subset, chunks, matrix, k, "keyword"), 3)
                for k in p3.K_VALUES
            },
        }

    # Q4: Abstention rates under default, more-willing (permissive), and less-willing (strict) prompts
    ans_sub = answerable[: p3.N_ABSTAIN]
    unans_sub = unanswerable[: p3.N_ABSTAIN]

    def eval_abstention_with_mode(mode_tag: str):
        def answer_variant(query: str, ret_chunks: list[dict]) -> dict:
            context = "\n\n".join(f"[{c['article_id']}]: {c['text']}" for c in ret_chunks)
            if mode_tag == "permissive":
                instruction = (
                    "Be helpful and more willing to answer even if partial information is present; "
                    f"make your best guess from the passages and only reply {p3.ABSTAIN!r} if completely unrelated."
                )
            elif mode_tag == "strict":
                instruction = (
                    "Be strict and less willing to guess: answer ONLY if 100% explicitly stated verbatim "
                    f"in the passages; otherwise reply {p3.ABSTAIN!r}."
                )
            else:
                instruction = (
                    f"Answer using ONLY the passages below. If they do not answer it, set \"answer\" to {p3.ABSTAIN!r}."
                )
            prompt = (
                f"{instruction}\n"
                'Return JSON with keys "answer" and "article_ids".\n\n'
                f"Passages:\n{context}\n\nQuestion: {query}"
            )
            reply = p2.chat([{"role": "user", "content": prompt}], temperature=0.0)
            parsed = p2.parse_json(reply)
            if not isinstance(parsed, dict):
                return {"answer": p3.ABSTAIN, "article_ids": []}
            return {
                "answer": str(parsed.get("answer", p3.ABSTAIN)),
                "article_ids": list(parsed.get("article_ids", [])),
            }

        def rate_for(subset):
            hits = 0
            for q in subset:
                ret_chunks = p3.retrieve(q["query"], chunks, matrix, k=3, method="embedding")
                res = answer_variant(q["query"], ret_chunks)
                if res.get("answer") == p3.ABSTAIN:
                    hits += 1
            return round(hits / len(subset), 3)

        return {
            "abstain_on_unanswerable_should": rate_for(unans_sub),
            "abstain_on_answerable_should_not": rate_for(ans_sub),
        }

    q4_rates = {
        "default": eval_abstention_with_mode("default"),
        "more_willing_permissive": eval_abstention_with_mode("permissive"),
        "less_willing_strict": eval_abstention_with_mode("strict"),
    }
    results["part3"] = {"recall": recall_table, "q4_abstention": q4_rates}

    # --- 5. Part 4 Memory & Q5 SUMMARY_WORDS ablation ---
    turns = p4.load_script()
    contexts_by_buf = {}
    for bsize in p4.BUFFER_SIZES:
        ctx = p4.run_conversation(turns, bsize).build_context(p4.NEXT_MESSAGE)
        contexts_by_buf[str(bsize)] = {
            "tokens": p4.count_tokens(ctx),
            "context": ctx,
        }

    summaries_by_word_limit = {}
    orig_words = p4.SUMMARY_WORDS
    for w_limit in (15, 25, 40, 80):
        p4.SUMMARY_WORDS = w_limit
        mem = p4.run_conversation(turns, 2)
        s_text = mem.summary.render()
        summaries_by_word_limit[str(w_limit)] = {
            "word_count": len(s_text.split()),
            "summary": s_text,
        }
    p4.SUMMARY_WORDS = orig_words

    results["part4"] = {
        "contexts_by_buffer_size": contexts_by_buf,
        "q5_summaries_by_word_limit": summaries_by_word_limit,
    }

    out_path = HERE / "hw2_experiment_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved HW2 experiment results to", out_path)


if __name__ == "__main__":
    main()
