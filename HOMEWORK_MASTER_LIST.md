# 📋 九大名校 AI Agents 课程课后作业全景整理（Master Problem & Assignment Specification）

本文档将 9 门经过逐字核验的顶尖高校 AI Agents 公开课的所有课后作业（Programming Assignments、Studios、Labs、Quizzes、Checkpoints、Short-Answer Questions）整理为统一的问题与任务规约清单。

---

## 1. CMU 11-768《AI Agents》（2026 秋，Graham Neubig & Daniel Fried）
- **官网**：https://cmu-agents.com/
- **成绩构成**：Assignments (45%: A1–A3) + Half-Semester Project (45%) + Lecture Highlights (10%: 22 讲中完成 20 讲课堂总结)

### 1.1 Assignment 1 · Build an Agent Harness（Due: 2026-09-14）
- **Starter Repo**：https://github.com/cmu-agents/assignment-1
- **Part 1 — Core Agent Loop & Progressive Skill Disclosure (35 pts)**
  - `TODO 1.1.a` (`src/agent/agents/base.py`): 实现 `Agent.build_prompt`，拼接系统提示词、工具描述、技能摘要与历史对话轨迹。
  - `TODO 1.1.b` (`src/agent/agents/code_agent.py`): 编写 `CodeAgent` 系统提示词与任务提示词模板。
  - `TODO 1.2` (`src/agent/agents/base.py`): 实现 `Agent.run` ReAct 循环（模型推理 $\rightarrow$ 解析工具调用 $\rightarrow$ 执行工具 $\rightarrow$ 追加观察结果 $\rightarrow$ 终止判定）。
  - `TODO 1.3` (`src/agent/agents/code_agent.py`): 实现 `CodeAgent.execute_tool_calls`，分派 `bash`、`read_file`、`write_file`、`edit_file` 等工具调用并捕获异常。
  - `TODO 1.4` (`src/agent/skills/loader.py` & `code_agent.py`): 实现技能渐进式披露（Progressive Disclosure）：启动时仅注入技能名称与简短描述，调用 `invoke_skill(skill_name)` 时按需加载完整 `SKILL.md` 指令；并在 `chess_app` 上完成 Bug 定位与修复。
- **Part 2 — Context Compaction (25 pts)**
  - `TODO 2.1` (`src/agent/agents/base.py`): 实现 `Agent.compact_context`，对早期多轮对话与冗长工具输出生成结构化摘要，保留系统消息与最近 $K$ 轮上下文。
  - `TODO 2.2` (`src/agent/agents/base.py`): 实现 `maybe_compact_context`，当累计 Token 超过阈值时自动触发上下文压缩。
  - **实验报告 (`token-usage-analysis.md`)**：在 SWE-bench 任务（如 `django__django-15368`）上对比开启/关闭上下文压缩的 Token 消耗、成本与解决率。
- **Part 3 — Chess Agent & Observation Engineering (40 pts)**
  - `TODO 3.1.a/b` (`chess_agent.py`, `chess_tools.py`): 实现 `play_move` 走子工具与非法走子重试机制。
  - `TODO 3.2` (**Observation A/B Experiment**): 开展 $2 \times 2$ 消融实验（棋盘表示：FEN vs ASCII Grid $\times$ 合法走子列表：提供 vs 不提供），撰写 `observation-experiment.md`。
  - `TODO 3.3.a/b`: 实现 `simulate_move` 推演工具（返回走子后的局面评估与对手威胁）。
  - `TODO 3.4`: 实现 `run_python` 沙箱执行工具，允许智能体编写 Python 脚本辅助计算。
  - `TODO 3.5`: 为 `ChessAgent` 挂载 `invoke_skill` 国际象棋战术技能库。

### 1.2 Assignment 2 · Evaluating a Data-Visualization Agent（Due: 2026-10-01）
- **Starter Repo**：https://github.com/cmu-agents/assignment-2
- **Part 1 — Baseline VLM Validator Analysis (20 pts)**
  - 在 `seed_runs/` 上评测基线多模态验证器对四类失败模式（`execution_failure`、`wrong_data`、`wrong_chart`、`hard_to_read`）的马修斯相关系数 (MCC)，诊断至少 2 类典型误判原因。
- **Part 2 — Improving the Validator (30 pts)**
  - 在 `validator/solution.py` 中重构 `validate(run)`，结合确定性代码/日志静态检查与多视角 VLM 提示策略，显著提升四类失败检测的 평균 MCC。
- **Part 3 — Authoring 5 Adversarial Tasks & Harbor Packaging (35 pts)**
  - 设计 5 个具有挑战性的数据可视化任务 (`tasks/<task_id>/task.json`)，对 15 条智能体运行轨迹进行人工 Ground-Truth 标注 (`labels.json`)。
  - 将 5 个任务打包为标准 Harbor 评测环境 (`harbor/tasks/<task_id>/`)，每个任务包含：
    - `task.toml`、`instruction.md`、`environment/Dockerfile` 及数据文件
    - `tests/test.sh` 与确定性状态验证脚本 `tests/test_state.py`
    - 标准参考解答 `solution/solve.sh`（得分 `1.0`）
    - 2 个变异体 (`mutants/`)：一个触发特定错误的负例（得分 `0.0`）与一个等价正例（得分 `1.0`）
- **Part 4 — Final Report (15 pts)**：撰写 `report.tex` / `report.md`。

### 1.3 Assignment 3 · Training（Due: 2026-10-29）
- 实现面向智能体轨迹的监督微调 (Rejection Sampling SFT) 与强化学习 (GRPO / PPO) 训练流程。

---

## 2. Columbia COMS 4995-15《Designing LLM Agents》（2026 秋，Shreya Havaldar）
- **官网**：https://designing-agents.github.io/

### 2.1 课堂编程实验 (In-Class Lab Activities)
- **Week 2 (`gradient_descent.py`)**：从零使用 NumPy 实现双层 MLP：One-hot 编码、ReLU/Softmax 前向传播、交叉熵损失、中心差分数值梯度校验、反向传播解析梯度与小批量 SGD 训练循环（Tasks 0–6）。
- **Week 3 (`beam_search.py`)**：基于 GPT-2 实现序列对数概率计算 `sequence_logprob`、Top-$k$ 单步扩展 `expand`、束搜索主循环 `beam_search`、长度归一化 `length_normalize(lp, length, alpha)` 与最优束选择 `best_beam`（Tasks 0–5）。
- **Week 4（2026-10-05 新发布：`autoprompting.py` & `gemini.py`）**：
  - `autoprompting.py`：实现自动提示词优化闭环（Automated Prompt Engineering）：编写初始 `GENERATOR_PROMPT`、定义结构化错误分类体系 `ERROR_TAXONOMY`、实现 LLM-as-a-Judge 评判函数 `judge()`、元提示词重写器 `rewrite()` 以及多轮迭代优化主循环 `autoprompt()`（Tasks 0–4）。
  - `gemini.py`：封装 Gemini API 调用、指数退避重试与 JSON 模式解析。

### 2.2 Homework 1: Data Processing, Tokenization, and Training（Due: 2026-10-05）
- **Starter Pack**：https://designing-agents.github.io/assets/HW1.zip
- **Part 1 (`part1_data_processing.py`, 25 pts)**：
  - 实现 1,000 条原始餐厅评论 (`data/unprocessed_data.jsonl`) 的完整清洗流水线：加载 JSONL、HTML 标签剥离与实体解码、Unicode NFKC 归一化與空白符折叠、正则 PII 脱敏（邮箱 `[EMAIL]`、电话 `[PHONE]`、URL `[URL]`）、启发式质量过滤（字数范围、重复字符、全大写垃圾评论、缺失评分或无效标签）、精确哈希去重与 MinHash/Jaccard 3-gram 近似去重。
  - 通过 `data/test_cases.json` 中全部 **52 个单元测试用例**，输出 `data/processed_data.jsonl`。
- **Part 2 (`part2_tokenizing.py`, 25 pts)**：
  - 对比 GPT-2 (BPE)、mBERT (WordPiece)、T5 (SentencePiece Unigram) 在标准英文、多语言餐饮术语、含 Emoji 评论及退化文本上的分词行为与 Fertility（每词 Token 数）。
  - 检验 Round-trip 编解码保真度 (`decode(encode(text)) == text`) 并定位 `<unk>` / 重音剥离导致的失真。
  - 从零实现并训练 **Byte-level BPE 分词器**（UTF-8 字节初始化 256 基础词表、相邻对频次统计、迭代合并规则学习、贪心按序编码与 UTF-8 解码）。
- **Part 3 (`part3_classifier.ipynb`, 30 pts)**：
  - 构建微型 Transformer 文本分类器（Token Embedding + Positional Embedding + 单层 Multi-Head Self-Attention + FFN + Mean Pooling + 5 类星级分类头）。
  - 在清洗后的评论数据集上对比不同 BPE 词表规模（300 vs 500 vs 2000）与隐藏层维度（`d_model=16` vs `d_model=128`）的训练损失曲线、训练集准确率与测试集准确率。
- **Short Answer Questions (`short_answer_Qs.md`, 20 pts)**：
  - Q1：对比三种预训练分词器在不同语料上的碎片化现象与原因。
  - Q2：分析 BPE 词表大小（300/500/2000）对序列长度、参数量及分类准确率的权衡。
  - Q3：诊断训练集与测试集准确率差距（过拟合成因与正则化改进方案）。
  - Q4：对比隐藏层维度 16 与 128 的表达能力、收敛速度与泛化表现。

### 2.3 Homework 2: Decoding, Prompting, Retrieval, and Memory（2026-10-05 新发布，Due: 2026-10-19）
- **Starter Pack**：https://designing-agents.github.io/assets/HW2.zip
- **Part 1 (`part1_decoding.py`, 15 pts)**：
  - 实现 Greedy 解码、Temperature 缩放、Top-$k$ 截断、Top-$p$ (Nucleus) 截断、重复惩罚 (`apply_repetition_penalty`)、组合采样器 `sample_next_token`、自回归生成循环 `generate` 以及生成多样性指标 `distinct_n`（Distinct-1 / Distinct-2）。
- **Part 2 (`part2_prompting.py`, 20 pts)**：
  - 实现带 SHA-256 磁盘缓存的 LLM 客户端，针对客服工单数据集 `data/tickets_labeled.jsonl`（50 条标注工单）实现四种提示策略：**Zero-shot**、**Few-shot**、**Chain-of-Thought (CoT)**、**Self-Consistency**（多路径采样 + 字段级多数投票），抽取结构化 JSON（`category`, `priority`, `requires_escalation`, `sentiment`）并实现 Schema 校验器 `validate_prediction` 与准确率评测。
- **Part 3 (`part3_retrieval.py`, 20 pts)**：
  - 基于知识库 `data/help_articles.jsonl`（30 篇文档）与评测查询集 `data/retrieval_queries.jsonl`（25 条含难负例查询），实现滑动窗口重叠分块 `chunk_document`、稠密向量索引 `build_index` / `search`、TF-IDF 基线检索 `tfidf_search`、计算 `recall_at_k`（`Recall@1`、`Recall@3`），并实现强制附带 `[doc_id]` 引用且对无法回答问题主动拒答的 `answer_query` 与引用核验器 `check_citation`。
- **Part 4 (`part4_memory.py`, 20 pts)**：
  - 在 40 轮长对话轨迹 `data/conversation.json`（含 8 道跨轮探针问题）上实现并评测三层记忆架构：
    1. `BufferMemory`：固定窗口滑动缓存最近 $k$ 轮对话
    2. `SummaryMemory`：增量式滚动摘要压缩历史对话
    3. `FactMemory`：抽取结构化实体事实三元组/键值对并支持相关性检索
    4. `MemoryAgent`：融合三层记忆回答跨度达 35+ 轮的用户偏好更新、否定修正与细节召回问题
- **Short Answer Questions (`short_answer_Qs.md`, 25 pts)**：5 道围绕解码退化、提示词策略性价比、稠密检索 vs 稀疏检索失效模式、记忆架构遗忘与冲突消解的深度简答题。

---

## 3. Columbia COMS W4995-009《Agentic Engineering》（2026 秋，Nick Gu）
- **官网**：https://course.nickgu.me | **仓库**：https://github.com/NickGuAI/agentic-engineering-course
- **Studio 01 — Delegation & The First Observable Run**：编写结构化委派卡 `delegation-card.md`，运行带完整 JSONL 轨迹记录的 Agent Harness 并通过 `verifier.py` 自动化校验。
- **Studio 02 — Context Window Stress Test & Memory Architecture**：
  - `part_a_stress.py`：量化测试随无关噪声文档增加时的上下文退化（"Lost in the Middle" 与精度下降曲线）。
  - `part_b_isolate_compress.py`：通过 Sub-agent 上下文隔离与结构化压缩摘要恢复准确率并降低主 Agent Token 峰值。
  - `part_c_memory.py`：实现跨会话持久化记忆读写。
- **Studio 03 — Agent Loops and Controlled Comparison**：
  - 在 `agent.py` 中实现顺序工具调用循环，并在相同搜索工具与预算下受控对比 **ReAct**（边想边搜）与 **Plan-and-Execute**（先规划后执行）在复杂技术调研问题上的表现。
- **Studio 05 — Build a Tool Server (MCP)（Due: 2026-10-16）**：
  - 基于本地数据集构建包含 3–8 个工具的 MCP Server，显式声明全部 4 个 MCP 工具注解（`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`），在代码层强制拒绝“删除全部数据”等破坏性请求，并构建含 10 个测试任务（单工具、多工具链、歧义边界、对抗性拒绝）的 `eval/tasks.jsonl`。

---

## 4. Stanford CS224V《Agentic AI》（2026 秋，Monica Lam）
- **官网**：https://web.stanford.edu/class/cs224v/
- **Homework 1: Finding the Patient — Agentic Literature Research for Rare-Disease Diagnosis（Due: 2026-10-06, 53 marks）**
  - 完成全部 12 个 TODO（1.1, 1.2, 3.1, 3.2, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 6.1），深入对比传统 RAG 与 **GRILL（Grounded Reasoning with Iterative Literature Lookup & Belief Ledger）** 在罕见病基因诊断队列及阴性对照患者上的鉴别诊断机制，输出 `todo_answers.json`。
- **Homework 2: Building Simple Agents in AOS & Formal Clinical Trial Matching with SatIR/VERDICT（Due: 2026-10-18, 110 marks）**
  - **Part 1 (57 marks, 16 TODOs)**：从底层 API 调用、工具选择、LangChain ReAct 智能体、Pydantic 结构化抽取（6a–6e）到 Agent Operating System (AOS) 与 SLIDERS 文档表的构建，输出 `answers.json`。
  - **Part 2 (53 marks, 6 TODOs: P2-1a/b, P2-2a, P2-3a/b, P2-4)**：分析 **SatIR / VERDICT** 如何将非结构化临床试验入组标准与患者病历转化为一阶逻辑约束并用 Z3 SMT 求解器进行可验证匹配，结合 GRILL 文献检索解决缺失检验指标推理，输出 `todo_answers.json`。

---

## 5. Stanford CS146S《The Modern Software Developer》（2026 秋，Mihail Eric）
- **官网**：https://themodernsoftware.dev | **仓库**：https://github.com/mihail911/modern-software-dev-assignments
- **Week 1 — Trace Dissection of a Real Coding Agent Session (100 pts)**：捕获真实多轮编码智能体会话 Trace，完成系统提示词核心指令批注、工具 Schema 设计盘点与缺陷分析、区分 `[OBSERVED]` 与 `[INFERRED]` 的行为剖析及架构改进提案。
- **Week 2 — Build an MCP Server for a Third-Party API (100 pts)**：实现包含 $\ge 3$ 个可组合工具、至少 1 个带 `dry_run` 保护刹车的写工具、严格 Schema 约束以及 OAuth 2.0 PKCE 令牌缓存与自动刷新机制的 MCP Server。
- **Week 3 — Author a Skill (100 pts)**：编写具备明确决策分支、配套脚本/模板资源的 `SKILL.md`，并通过 3 个正例 Prompt 与 1 个近邻负例 Prompt 验证触发边界与端到端执行效果。

---

## 6. Stanford CS329A《Self-Improving AI Agents》（Aakanksha Chowdhery & Azalia Mirhoseini）
- **官网**：https://cs329a.stanford.edu/
- **Homework 1 — Test-Time Compute Scaling, Verification & Execution Feedback（Due: 2026-10-13）**：实现无偏 `pass@k` 估计器、Majority Voting、基于验证器打分的 Best-of-$N$ 重排序以及结合单元测试异常回溯的 Self-Debugging 循环。
- **Homework 2 — Multi-Step Reasoning/Planning (LATS/ADaPT) & Train-Time RL Scaling (STaR/GRPO)（Due: 2026-10-22）**：实现结合蒙特卡洛树搜索与语言自反思的 LATS 算法、按需递归分解的 ADaPT 算法，以及 STaR 轨迹过滤与 GRPO（Group Relative Policy Optimization）强化学习更新。
- **Homework 3 — Agentic Search (Search-o1) & Kernel/Code Optimization Agents（Due: 2026-11-07）**：实现推理内嵌检索与文档压缩的 Search-o1 智能体，以及基于性能剖析反馈（Runtime Profiling + 数值等价性校验）的算子/代码自动优化智能体。

---

## 7. Berkeley CS294-196《Advanced LLM Agents》（2025 春 / 2024 秋，Dawn Song）
- **官网**：https://llmagents-learning.org/sp25
- **Weekly Quizzes 1–12**：涵盖智能体架构、推理搜索、工具调用、记忆系统、代码智能体、Lean 4 形式化验证、多模态/Web 智能体、多智能体协作、强化学习后训练、智能体安全与抗提示注入、基准评测与复合 AI 系统的 12 套周测验与标准解答。
- **Lean 4 Verifiable Code & Theorem Proving Labs (Lab 1 & Lab 2)** + **Foundation Labs 1–3**：实现自然语言到 Lean 4 形式化规约的自动形式化（Autoformalization）、基于编译器报错反馈的定理证明修复循环，以及 ReAct / RAG / Reflexion 基础实验。
- **Written Technical Article**：撰写 500+ 词的智能体形式化验证与自进化技术综述文章。

---

## 8. Harvard CS249r《Architecture 2.0: Agentic AI for Computer Systems Design》（2025 秋，Vijay Janapa Reddi）
- **官网**：https://harvard-edge.github.io/cs249r_fall2025
- **Checkpoint 1 (Literature Survey)**：横跨硬件/RTL 设计智能体、编译器/GPU 算子优化智能体、云与操作系统运维智能体、以及面向智能体负载的系统架构（PagedAttention/SGLang/KV-Cache 共享）四大领域的 25 篇核心论文分类与数据抽取规约。
- **Checkpoint 2 (Data-Driven Exercise)**：对 25 篇系统论文进行跨栈定量指标提取与统计建模（Token 放大倍数、验证器耗时占比、KV-Cache 命中率、`pass@k` 加速比）。
- **Checkpoint 3 & Final Integrated Survey Paper**：完整撰写《Architecture 2.0: Agentic AI for Computer Systems Design and Systems for Agentic AI》学术综述论文。

---

## 9. 上海交通大学《动手学大模型智能体》(Hands-on AI Agent, 张伟楠等)
- **官网**：https://haa.boyuai.com | **仓库**：https://github.com/boyu-ai/Hands-on-AA
- **第一部分：智能体基础（第 1–4 章）**：感知-行动闭环、因果语言模型与 KV Cache 解码、提示词工程（CoT / Self-Consistency）、智能体评测指标与调试。
- **第二部分：核心模块与架构（第 5–8 章）**：ReAct 与 Plan-and-Solve 架构、短期/长期记忆与混合 RAG、函数调用与沙箱执行、思维树 (ToT) 与蒙特卡洛树搜索 (MCTS)。
- **第三部分：智能体训练与微调（第 9–11 章）**：智能体轨迹掩码监督微调 (SFT)、低秩自适应 (LoRA) 与 INT8/INT4 量化、DPO 与 GRPO 强化学习对齐。
- **第四部分：前沿专题（第 12–15 章）**：多模态 GUI 视觉接地智能体、SOP 多智能体协作与辩论系统、提示注入防御与权限策略引擎、MCP 与 A2A 智能体通信协议。
