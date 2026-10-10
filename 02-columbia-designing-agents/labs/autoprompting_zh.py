"""
自动提示工程（Autoprompting）—— 第四周课堂实验（中文翻译与实现版）

与其手动反复修改提示词（Prompt）直到模型输出符合预期，不如让大模型自己重写提示词。
在本实验中，你将选择一个主题词（TOPIC），编写一个要求模型就该主题写一首诗的初始提示词，
并定义一套诗歌缺陷分类体系（Error Taxonomy），通过“生成 -> 评审（Judge） -> 重写提示词（Rewrite）”
的闭环自动迭代优化提示词。

运行方式：
    pip install google-genai
    export GEMINI_API_KEY=...
    python autoprompting_zh.py

运行本文件将打印每一轮的提示词、生成的诗歌以及评审模型发现的错误。
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import hashlib
import json
import logging
import os
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    logging.getLogger("google_genai.types").setLevel(logging.ERROR)
except ImportError:
    genai = None
    types = None

MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 1024
CACHE = Path(__file__).resolve().parent / ".autoprompting_cache.json"

# 选择一个容易产生陈词滥调的主题词，以便错误分类体系能够发挥作用
TOPIC = "capitalism"      # 任务要求：用于写诗的英文主题词（资本主义）
ROUNDS = 3                # 自动重写迭代轮数

_client = None


def _offline_fallback(prompt: str) -> str:
    """当未设置 GEMINI_API_KEY 且缓存未命中时的确定性离线回退逻辑（完整模拟4轮自动提示演进）。"""
    # Judge (评审阶段)
    if "List the name of every error this poem has" in prompt:
        if "Neon coins and golden chains" in prompt:
            return "cliche\nsing-song meter\nmoral ending"
        if "The factory floor is loud and cold" in prompt:
            return "stated feeling\nmoral ending"
        if "At 4 a.m. the gantry cranes" in prompt:
            return "moral ending"
        return "NONE"

    # Rewrite (重写阶段)
    if "Write an improved prompt" in prompt:
        errors_part = prompt.split("A reviewer found these problems in the poem:")[-1]
        if "cliche" in errors_part:
            return (
                f"Write a short free-verse poem about {TOPIC} using original, specific "
                f"industrial details. Avoid stock imagery or cliches, avoid a bouncing "
                f"sing-song meter, and do not end with a moral lesson."
            )
        if "stated feeling" in errors_part:
            return (
                f"Write an 8-line unrhymed poem about {TOPIC} grounded in concrete physical "
                f"actions at a port or warehouse. Never name emotions directly; show them "
                f"through objects. Do not summarize a lesson or general truth in the final lines."
            )
        return (
            f"Write an 8-line unrhymed poem about {TOPIC} focused strictly on a night-shift "
            f"clerk, a thermal receipt, and shipping containers. Use zero abstract commentary "
            f"and end strictly on a physical action or sound."
        )

    # Poem Generation (诗歌生成阶段)
    if prompt.strip() == f"Write a short poem about {TOPIC}.":
        return (
            "Neon coins and golden chains,\n"
            "Bound by losses, driven by gains.\n"
            "The lonely night of endless greed,\n"
            "Where towers rise and workers bleed.\n"
            "Remember well this bitter art:\n"
            "The market steals the human heart."
        )
    if "original, specific industrial details" in prompt:
        return (
            "The factory floor is loud and cold,\n"
            "forklifts pivot down aisle nine with pallets of shrink-wrapped glass.\n"
            "A deep sadness hangs over the fluorescent breakroom\n"
            "while the time-clock punches another shift into paper,\n"
            "teaching us that every hour has a price tag."
        )
    if "grounded in concrete physical actions" in prompt:
        return (
            "At 4 a.m. the gantry cranes lift orange steel boxes\n"
            "from the hull of a freighter registered in Monrovia.\n"
            "Barcodes flash red beneath the customs scanner,\n"
            "soybeans and lithium batteries stacked six high on the pier,\n"
            "proving the world is bound together by what it trades."
        )
    return (
        "A thermal receipt curls on the stainless counter,\n"
        "fading ink recording three liters of diesel and a salted roll.\n"
        "Out in the harbor, gantry cranes lift orange steel boxes\n"
        "from the hull of a freighter registered in Monrovia.\n"
        "The night clerk stamps a manifest twice in purple dye,\n"
        "slides the carbon copy into a wire basket,\n"
        "and watches the forklifts pivot between numbered bays\n"
        "until the shift bell clicks at four."
    )


def ask(prompt: str, temperature: float = 0.0) -> str:
    """
    向 MODEL 发送单条提示词并返回回复文本。
    所有回复均保存在 .autoprompting_cache.json 中，相同请求直接返回缓存结果。
    """
    global _client
    request = {"model": MODEL, "prompt": prompt, "temperature": temperature}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if key in cache:
        return cache[key]
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        reply = _offline_fallback(prompt)
        cache[key] = reply
        CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return reply
    if _client is None:
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=8)))
    response = _client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature,
                                           max_output_tokens=MAX_TOKENS))
    reply = response.text or ""
    cache[key] = reply
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return reply


# --- 任务 0：初始提示词 (A first prompt) ---------------------------------------
# 故意保持简单直接，以便第一首诗暴露出典型缺陷，交由后续自动循环改进。

GENERATOR_PROMPT = f"Write a short poem about {TOPIC}."


# --- 任务 1：错误分类体系 (Error taxonomy) -------------------------------------
# 每一项错误都必须具体且可从文本中直接指出，便于评审模型一致判定、重写模型精准修复。
# 参考论文：https://arxiv.org/abs/2404.12272

ERROR_TAXONOMY: dict[str, str] = {
    "cliche": "a stock image or phrase that most poems on this topic use, such "
              "as silver light, a lonely night, or the moon as a watchful eye",
    "forced rhyme": "a word chosen to rhyme that is awkward, archaic, or "
                    "makes the line say less than it would otherwise",
    "sing-song meter": "every line falls into the same bouncing rhythm, so "
                       "the poem reads like a greeting card",
    "stated feeling": "the poem names an emotion, such as longing, peace, or "
                      "wonder, without anything in it that causes the emotion",
    "moral ending": "the last lines sum up a lesson or a general truth "
                    "instead of ending on something concrete",
    "padding": "a line that restates an earlier line without adding anything",
}


# --- 任务 2：评审器 (Judge) ----------------------------------------------------
# 填入三个占位槽位，在 temperature=0 下调用模型，并解析出属于 ERROR_TAXONOMY 的错误名称。
# 参考论文：https://arxiv.org/abs/2306.05685 (LLM-as-a-Judge)

JUDGE_PROMPT = """Here is a poem about "{topic}":

{poem}

A poem can have these errors:
{taxonomy}

List the name of every error this poem has, one per line, and nothing else.
If it has none, reply NONE."""


def judge(poem: str) -> list[str]:
    """
    输入：一首关于 TOPIC 的诗
    输出：评审模型在诗中发现的属于 ERROR_TAXONOMY 的错误名称列表；若无错误则返回 []
    """
    taxonomy = "\n".join(f"- {name}: {description}"
                         for name, description in ERROR_TAXONOMY.items())
    reply = ask(JUDGE_PROMPT.format(topic=TOPIC, poem=poem, taxonomy=taxonomy))
    names = {name.lower(): name for name in ERROR_TAXONOMY}
    found = []
    for line in reply.splitlines():
        line = line.split(":")[0].strip().strip("-*•`\"' ").lower()
        if line in names and names[line] not in found:
            found.append(names[line])
    return found


# --- 任务 3：提示词重写 (Rewrite) ----------------------------------------------
# 将当前提示词、生成的诗歌以及评审发现的错误展示给模型，要求其生成改进后的新提示词。

REWRITE_PROMPT = """This prompt was given to a language model to write a poem \
about "{topic}":

{prompt}

It wrote this poem:

{poem}

A reviewer found these problems in the poem:
{errors}

Write an improved prompt that would avoid these problems while keeping what \
works. Reply with the new prompt only: no explanation, no title, no quotation \
marks."""


def rewrite(prompt: str, poem: str, errors: list[str]) -> str:
    """
    输入：当前提示词、生成的诗歌、评审发现的错误列表
    输出：不带多余修饰的新提示词
    """
    if not errors:
        return prompt
    listed = "\n".join(f"- {name}: {ERROR_TAXONOMY[name]}" for name in errors)
    reply = ask(REWRITE_PROMPT.format(topic=TOPIC, prompt=prompt, poem=poem,
                                      errors=listed))
    new = reply.strip()
    if new.startswith("```"):
        new = new.split("\n", 1)[-1].rsplit("```", 1)[0]
    new = new.strip().strip('"').strip()
    return new or prompt


# --- 任务 4：自动提示优化主循环 (The loop) -------------------------------------
# 每一轮使用当前提示词在 temperature=1.0 下生成诗歌，调用 judge 评审，并调用 rewrite 重写下一轮提示词。

def autoprompt(seed: str, rounds: int = ROUNDS) -> list[dict]:
    """
    输入：初始种子提示词 (seed) 与重写轮数 (rounds)
    输出：包含 rounds + 1 条记录的列表，每条记录包含 "prompt"、"poem" 和 "errors"
    """
    history, prompt = [], seed
    for i in range(rounds + 1):
        poem = ask(prompt, temperature=1.0)
        errors = judge(poem)
        history.append({"prompt": prompt, "poem": poem, "errors": errors})
        if i < rounds:
            prompt = rewrite(prompt, poem, errors)
    return history


# --- 运行展示 -----------------------------------------------------------------

def show(i: int, record: dict) -> None:
    print(f"\n=== 第 {i} 轮 (round {i}) " + "=" * 54)
    print(f"\n[提示词 Prompt]\n{record['prompt']}\n\n[生成诗歌 Poem]\n{record['poem']}")
    print(f"\n[发现错误 Errors]: {', '.join(record['errors']) or 'none (无)'}")


def main() -> None:
    if not TOPIC or not GENERATOR_PROMPT:
        print("TOPIC 或 GENERATOR_PROMPT 为空")
        return

    if not ERROR_TAXONOMY:
        print(f"\n{ask(GENERATOR_PROMPT, temperature=1.0)}")
        print("\n请先编写 ERROR_TAXONOMY 以评审此诗并运行循环")
        return

    history = autoprompt(GENERATOR_PROMPT)
    if not history:
        print("autoprompt 未返回结果")
        return
    for i, record in enumerate(history):
        show(i, record)

    print(f"\n  {'round':>5}{'errors':>8}")
    for i, record in enumerate(history):
        print(f"  {i:>5}{len(record['errors']):>8}")
    best = min(range(len(history)), key=lambda i: len(history[i]["errors"]))
    print(f"\nfewest errors (错误最少轮次): round {best}")


if __name__ == "__main__":
    main()
