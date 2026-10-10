# 🎓 九大名校 AI Agents 课程课后作业全景清单与完整实现 (AI Agents Courses Homework & Solutions)

> **核验基线**：2026-10-05 逐字核验 + 2026-10-10 最新增量同步（含 Columbia Week 4 新发布的 `HW2`、`autoprompting.py`、`gemini.py` 中文翻译与实现，以及 CMU 11-768 第 12–14 讲进度核验）。
>
> **跨平台同步中心**：
> - 🐙 **GitHub 仓库**：[https://github.com/yixinzhangagent/ai-agents-courses-homework](https://github.com/yixinzhangagent/ai-agents-courses-homework)
> - 📌 **GitHub Master Issue #1**：[https://github.com/yixinzhangagent/ai-agents-courses-homework/issues/1](https://github.com/yixinzhangagent/ai-agents-courses-homework/issues/1)
> - 📝 **Notion 追踪页面**：[🎓 九大名校 AI Agents 课程课后作业全景清单与完成追踪 (2026 秋)](https://app.notion.com/p/3f5567e95a6a81b8bd81d2a99b53c4de?pvs=204)
> - ✅ **Todoist 专项项目**：`九大名校 AI Agents 课后作业全集` (`id=6hj9w8c6PwWR8CRr`)

---

## 🗺️ 仓库目录结构与九大课程映射

| 目录 | 学校与课程编号 | 课程名称 | 授课教师 / 学期 | 官网链接 | 核心作业与实验内容 |
|---|---|---|---|---|---|
| [`01-cmu-11-768/`](./01-cmu-11-768/) | **CMU 11-768** | **AI Agents** | Graham Neubig, Daniel Fried (2026 秋) | [cmu-agents.com](https://cmu-agents.com/) | **Assignment 1** (Agent Harness, Context Compaction, Chess Agent)、**Assignment 2** (Data-Vis VLM Validator & 5 Harbor Tasks)、**Assignment 3** 参考实现、第 12–14 讲中文讲义 |
| [`02-columbia-designing-agents/`](./02-columbia-designing-agents/) | **Columbia COMS 4995-15** | **Designing LLM Agents** | Shreya Havaldar (2026 秋) | [designing-agents.github.io](https://designing-agents.github.io/) | **Week 2–4 Labs** (`gradient_descent.py`, `beam_search.py`, `autoprompting.py`, `gemini.py` + 中文翻译)、**HW1** (数据清洗 52/52 测试、BPE、Transformer 分类器 + 简答)、**HW2** (10-05 新发：解码、提示词、RAG 检索、三层记忆 + 简答) |
| [`03-columbia-agentic-engineering/`](./03-columbia-agentic-engineering/) | **Columbia COMS W4995-009** | **Agentic Engineering** | Nick Gu (2026 秋) | [course.nickgu.me](https://course.nickgu.me) | **Studio 01** (Delegation & Observable Harness)、**Studio 02** (Context Stress Test & Memory)、**Studio 03** (ReAct vs Plan-and-Execute)、**Studio 05** (本地数据 MCP Server + 4 Annotations + 破坏性操作代码拦截 + 10-Task Eval) |
| [`04-stanford-cs224v/`](./04-stanford-cs224v/) | **Stanford CS224V** | **Agentic AI** | Monica Lam (2026 秋) | [web.stanford.edu/class/cs224v](https://web.stanford.edu/class/cs224v/) | **HW1** (罕见病文献诊断智能体 RAG vs GRILL 信念账本 12 TODOs)、**HW2 Part 1** (Building Simple Agents & AOS 16 TODOs)、**HW2 Part 2** (SatIR/VERDICT 临床试验形式化匹配 6 TODOs) |
| [`05-stanford-cs146s/`](./05-stanford-cs146s/) | **Stanford CS146S** | **The Modern Software Developer** | Mihail Eric (2026 秋) | [themodernsoftware.dev](https://themodernsoftware.dev) | **Week 1** (Claude Code 会话 Trace 解剖)、**Week 2** (含 OAuth 2.0 与 `dry_run` 写保护的第三方 API MCP Server)、**Week 3** (可复用 Agent Skill 编写与正负触发测试) |
| [`06-stanford-cs329a/`](./06-stanford-cs329a/) | **Stanford CS329A** | **Self-Improving AI Agents** | Aakanksha Chowdhery, Azalia Mirhoseini | [cs329a.stanford.edu](https://cs329a.stanford.edu/) | **HW1** (Test-Time Compute Scaling, Verifier & Self-Debugging)、**HW2** (LATS 树搜索规划、ADaPT & STaR/GRPO 强化学习)、**HW3** (Search-o1 迭代检索推理 & 算子/代码性能优化智能体) |
| [`07-berkeley-cs294/`](./07-berkeley-cs294/) | **Berkeley CS294-196** | **Advanced LLM Agents** | Dawn Song (2025 春 / 2024 秋) | [llmagents-learning.org/sp25](https://llmagents-learning.org/sp25) | **Quizzes 1–12** 全套解析、**Lean 4 形式化验证与自动定理证明智能体 Lab**、**Foundation Labs 1–3**、500+ 词技术综述文章 |
| [`08-harvard-cs249r/`](./08-harvard-cs249r/) | **Harvard CS249r** | **Architecture 2.0: Agentic AI for Computer Systems Design** | Vijay Janapa Reddi (2025 秋) | [harvard-edge.github.io/cs249r_fall2025](https://harvard-edge.github.io/cs249r_fall2025) | **Checkpoint 1** (25 篇系统/硬件智能体文献综述)、**Checkpoint 2** (跨论文定量指标分析脚本与图表)、**Checkpoint 3 & Final Survey Paper** |
| [`09-sjtu-hands-on-aa/`](./09-sjtu-hands-on-aa/) | **上海交通大学 SJTU** | **《动手学大模型智能体》** | 张伟楠等 (中文开源) | [haa.boyuai.com](https://haa.boyuai.com) | **Part 1–4 共 15 章**可运行智能体算法模块与全套单元测试（ReAct、Plan-and-Solve、RAG/Memory、Tool Calling、ToT/MCTS、SFT、LoRA、DPO/GRPO、多模态 GUI、MAS、安全对齐、MCP/A2A 协议） |

> **关于 MIT 的核验说明**：2026-10-05 已逐字核验，MIT 目前没有完整公开的对标 AI Agents 专业课，最接近的是 Paul Liang 多模态课程（MIT 6.S986 / 6.8611）中包含的单讲 Agent 内容，另有收费职业培训课，因此不单列虚假作业目录。

---

## 🔥 2026-10-05 以来最新课程动态与补齐说明

1. **Columbia《Designing LLM Agents》Week 4（10-05 周一晚发布）**：
   - **新增课件**：*Week 4 — Prompting, Reasoning, & Memory*（中文完整翻译与精读笔记见 [`02-columbia-designing-agents/labs/Week4_Prompting_Reasoning_Memory_zh.md`](./02-columbia-designing-agents/labs/Week4_Prompting_Reasoning_Memory_zh.md)）。
   - **新增课堂代码**：[`autoprompting.py`](./02-columbia-designing-agents/labs/autoprompting.py)（含中文翻译详注版 [`autoprompting_zh.py`](./02-columbia-designing-agents/labs/autoprompting_zh.py)）与 [`gemini.py`](./02-columbia-designing-agents/labs/gemini.py)（含中文翻译详注版 [`gemini_zh.py`](./02-columbia-designing-agents/labs/gemini_zh.py)）。
   - **新增课后作业 HW2**：[`02-columbia-designing-agents/HW2/`](./02-columbia-designing-agents/HW2/)（涵盖解码策略对比、结构化提示词工程、稠密向量 vs TF-IDF 检索与带引用 Grounded QA、三层对话记忆架构及 5 道深度简答题）。
2. **CMU 11-768《AI Agents》第 12–14 讲核验（10-06 & 10-08）**：
   - 核验了官网课程表（`schedule.ts`）：第 12 讲 *Long Context and Context Management* 已发布讲义 PDF；第 13 讲（10-06，*Agentic Evolution and Continual Learning*，Freda Shi）与第 14 讲（10-08，*GUI and Web Agents*）已完成课堂专题中文梳理与翻译，详见 [`01-cmu-11-768/lectures_and_a3/CMU_Lectures_12_13_14_zh.md`](./01-cmu-11-768/lectures_and_a3/CMU_Lectures_12_13_14_zh.md)。

---

## 🧪 快速运行与验证指南

每个课程子目录均包含独立的 `README.md`、测试套件与运行说明。你可以在各子目录下直接运行 `pytest` 或对应的评测脚本验证作业结果。
