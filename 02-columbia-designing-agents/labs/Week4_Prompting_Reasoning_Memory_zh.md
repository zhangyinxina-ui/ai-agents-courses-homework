# 哥伦比亚大学 COMS 4995-15《Designing LLM Agents》第 4 周讲义与实验完整中文翻译

- **课程名称**：COMS 4995-15 Designing LLM Agents（设计大语言模型智能体，2026 年秋季）
- **授课教师**：Shreya Havaldar（哥伦比亚大学）
- **第 4 周主题**：**Prompting, Reasoning, & Memory（提示工程、推理技术与记忆管理）**
- **原始讲义下载**：[`labs/week4.pdf`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/week4.pdf)（共 56 页幻灯片）
- **配套课堂代码**：
  - 英文原版实现：[`labs/autoprompting.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/autoprompting.py)、[`labs/gemini.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/gemini.py)
  - 中文翻译与注释版：[`labs/autoprompting_zh.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/autoprompting_zh.py)、[`labs/gemini_zh.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/gemini_zh.py)

---

## 目录

1. [第一部分：提示工程（Prompting, Slides 1–22）](#1-第一部分提示工程prompting-slides-122)
2. [第二部分：推理技术与推理模型（Reasoning Techniques & Reasoning Models, Slides 23–44）](#2-第二部分推理技术与推理模型reasoning-techniques--reasoning-models-slides-2344)
3. [第三部分：记忆管理（Memory Management, Slides 45–56）](#3-第三部分记忆管理memory-management-slides-4556)
4. [第四部分：课堂讨论题解答（Slide 56 Discussion）](#4-第四部分课堂讨论题解答slide-56-discussion)
5. [第五部分：课堂编程实验详解（Autoprompting & Gemini API）](#5-第五部分课堂编程实验详解autoprompting--gemini-api)

---

## 1. 第一部分：提示工程（Prompting, Slides 1–22）

### Slide 1–3：课程公告与今日议程（Announcements & Today's Lecture）
- **课程通知（Slide 2）**：
  - 作业 1（HW 1）于今晚美东时间 11:59 pm 截止（每学期共有 3 天迟交额度）。
  - 作业 2（HW 2）将于明天正式发布。
- **今日课程三大模块（Slide 3）**：
  1. **提示工程（Prompting）**
  2. **推理技术（Reasoning techniques）**
  3. **记忆管理（Memory management）**

### Slide 4–5：核心视角——提示词不是“问题”，而是“形式化规范”（Prompts as Specifications）
- **核心认知框架（Slide 4）**：
  > **“Prompts are not questions; they are specifications.”（提示词不是向模型随口提问的问题，而是工程规范/技术规格说明书。）**
- **作为规格说明书（Specification）的五大核心要素（Slide 5）**：
  1. **目标（Goal）**：明确什么是“成功”（What success means）。
  2. **输入（Inputs）**：提供了哪些信息，缺失了哪些信息（What is provided; what is missing）。
  3. **约束条件（Constraints）**：必须做/绝对不能做的事项（must/never）、策略规范（policy）、语言风格（style）、长度限制（length）。
  4. **输出格式（Output format）**：严格的数据模式（schema）、章节结构（sections）、确定性的格式化要求（deterministic formatting）。
  5. **验收测试（Acceptance tests）**：提供 2–5 个正确输出与错误输出的对比示例（2–5 examples of correct/incorrect outputs）。

### Slide 6–7：提示工程与标准提示词布局（Prompt Engineering & Typical Layout）
- **定义（Slide 6）**：
  - 提示词（Prompts）始终包含传递给 LLM 以完成特定任务的**指令（instructions）**和**上下文（context）**。
  - **提示工程（Prompt engineering）** 是开发、调试并优化这些提示词的工程实践。
- **标准提示词结构布局（Typical prompt layout, Slide 7）**：
  使用一致的模块化分块（consistent blocks）以减少歧义并提升可调试性：
  - **角色/任务（Role/Task）**：“你是……你的任务是……”（`You are ... your task is ...`）
  - **规则（Rules）**：使用项目符号列表（bullet list），直白清晰地列出规则。
  - **上下文（Context）**：提供任务所需的任何事实或背景数据。
  - **输出格式（Output format）**：精确的 Schema 定义 + “不要输出任何额外文本”（`exact schema + "no extra text"`）。
  - **示例（Examples）**：正例与反例（如果需要的话）。

### Slide 8–10：提示词设计三大法则（Design Tips）
1. **从简单开始（Start simple, Slide 8）**：
   - 提示工程是一个**迭代过程（iterative process）**！
   - 实验 $\rightarrow$ 微调提示词 $\rightarrow$ 重复验证（Experiment, tweak the prompt, repeat）。
   - **谨记三大黄金准则**：**具体性（Specificity）、简洁性（Simplicity）、精炼性（Conciseness）**。
2. **将复杂任务拆解为子任务（Break things down into subtasks, Slide 9）**：
   - LLM 通常不擅长同时做好多件复杂的事情。
   - 拆解子任务有助于保持每个提示词简单且具体。
   - 针对 LLM 表现吃力的组件，进行迭代式细化拆解。
3. **明确说明任务与期望输出（Be clear about the task and expected output, Slide 10）**：
   - 提示词描述越清晰具体，结果通常越好。
   - 但要注意：**过多不必要的细节（unnecessary details）反而会降低结果质量**。
   - 提供示例（Examples）几乎总是有益的。

### Slide 11–14：正反案例对比（Bad Prompt vs. Better Prompt）

#### 案例一：概念解释（Slides 11–12）
| 维度 | 糟糕的提示词（A bad prompt, Slide 11） | 优秀的提示词（A better prompt, Slide 12） |
| :--- | :--- | :--- |
| **英文原文** | *"Explain the concept prompt engineering. Keep the explanation short, only a few sentences, and don't be too descriptive"* | *"Use 2-3 sentences to explain the concept of prompt engineering to a high school student."* |
| **中文翻译** | “解释提示工程的概念。保持解释简短，只用几句话，而且不要太详细。” | “用 2-3 句话向一名高中生解释什么是提示工程。” |
| **工程诊断** | ❌ **解释长度模糊**（“几句话”不明确）<br>❌ **解释风格模糊**（“不要太详细”是负向且主观的描述） | ✅ **明确量化长度**（2–3 句话）<br>✅ **明确目标受众与语言风格**（面向高中生） |

#### 案例二：电影推荐智能体（Slides 13–14）
| 维度 | 糟糕的提示词（A bad prompt, Slide 13） | 优秀的提示词（A better prompt, Slide 14） |
| :--- | :--- | :--- |
| **英文原文** | *"You are an agent that recommends movies to a customer. DO NOT ASK FOR INTERESTS. DO NOT ASK FOR PERSONAL INFORMATION."* | *"You are an agent that recommends movies to a customer. The agent should recommend a movie from the top global trending movies. It should not ask users for their preferences or personal information. If the agent doesn't have a movie to recommend, it should respond 'Sorry, no movie available'"* |
| **中文翻译** | “你是一个向客户推荐电影的智能体。不要询问兴趣爱好。不要询问个人信息。” | “你是一个向客户推荐电影的智能体。智能体应当从全球热门趋势电影中推荐一部电影。它不应询问用户的偏好或个人信息。如果智能体没有可推荐的电影，应当回复‘Sorry, no movie available’。” |
| **工程诊断** | ❌ **缺乏正向输出约束**（只说了不能做什么，没说该依据什么推荐）<br>❌ **未定义边界情况（Edge cases）**（没有电影可推荐时怎么办？） | ✅ **明确输出约束**（从全球热门趋势电影中推荐，不询问偏好或隐私）<br>✅ **明确边界情况回退输出**（无电影时固定回复 `"Sorry, no movie available"`） |

### Slide 15–17：软提示（Soft Prompting）与硬提示（Hard Prompting）
- **什么是软提示（Soft Prompting, Slide 15）**：
  软提示是一种**参数高效（parameter-efficient）**的微调技术，它使用**可学习的连续嵌入向量（learned embeddings）**而非人类可读的自然语言文本来引导 LLM：
  1. 在输入词元嵌入序列（input embedding sequence）前端直接拼接**可训练向量（trainable vectors，也称“虚拟词元 virtual tokens”）**。
  2. 在训练过程中保持**核心 LLM 权重完全冻结（completely frozen）**。
  3. **梯度优化（Gradient Optimization）**：在特定任务训练集上通过反向传播仅更新这一小部分提示参数。
- **硬提示 vs. 软提示对比（Slides 16–17）**：
  - **硬提示（Hard Prompting）**：例如在输入 `"I will definitely watch it again."` 后拼接固定文本模板 `"It is [MASK]"`。
    - **优点**：完全透明，人类易于阅读和调试（Fully transparent and easy for humans to read + debug）。
  - **软提示（Soft Prompting）**：在输入词元嵌入前拼接若干可调嵌入向量（Tunable Embeddings）。
    - **优缺点**：对人类完全不透明且不可读（Opaque and completely unreadable to humans）。在高度专用的特定任务上效果很好，但在当今通用大模型时代已不太流行（not popular nowadays）。

### Slide 18–22：提示模板与 API 工程最佳实践（Engineering Best Practices）
- **常见基础任务的提示模板（Prompting templates, Slide 18）**：
  大多数基础 LLM 任务都有成熟的提示模板（如文本摘要 Text Summarization、信息抽取 Information Extraction、问答 Question Answering、文本分类 Text Classification、代码生成 Code Generation）。
- **最佳实践 1：指数退避重试（Exponential Backoff, Slide 19）**：
  当客户端调用服务端遭遇限流（HTTP 429）或瞬时失败时，不应立即密集重试，而应按指数增长延迟等待时间（如等待 $100\text{ ms} \rightarrow 200\text{ ms} \rightarrow 400\text{ ms} \rightarrow \dots$）直至请求成功。
- **最佳实践 2：提示缓存（Prompt Caching, Slides 20–21）**：
  - 对重复查询优先检索本地缓存（Cache），命中（Hit）则直接返回响应 `r`，未命中（Miss）才调用 LLM 并将响应写入缓存。
  - **代码实现要点（Slide 21）**：使用 `hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()` 将任意请求字典（模型名、提示词、温度等）映射为唯一的 64 位十六进制哈希字符串作为缓存键（Key）。
- **最佳实践 3：API 密钥安全（API Key Safety, Slide 22）**：
  - 在终端设置环境变量：`export GEMINI_API_KEY=<your_api_key>`
  - 在代码中读取：`api_key = os.environ["GEMINI_API_KEY"]`
  - 将 API 密钥存储为环境变量是防止凭据意外泄露的行业标准做法，让你可以安全地分享或提交代码库而不会暴露私密密钥。

---

## 2. 第二部分：推理技术与推理模型（Reasoning Techniques & Reasoning Models, Slides 23–44）

### Slide 24–26：什么是推理？何时需要推理提示？（What is Reasoning?）
- **定义（Slides 24–25）**：
  - **LLM 中的推理（Reasoning in LLMs）**：模型通过将复杂的多步问题拆解为**中间的、逻辑化的、顺序的步骤（intermediate, logical, and sequential steps）**来解决复杂问题的能力。
  - **提示词推理（Reasoning in prompts）**：拿来任意一个通用 LLM，通过特定的提示方式激发（elicit）其推理能力。
  - **推理模型（Reasoning models）**：专门经过微调（fine-tuned），在生成最终输出之前先进行**内部推理（internally reason）**的 LLM（如 OpenAI o1/o3、DeepSeek-R1、Gemini Thinking 等）。
- **推理提示的适用边界（When reasoning prompts help, Slide 26）**：
  - **通常有帮助的场景**：
    - 多步算术计算（Multi-step arithmetic）
    - 多约束决策（Multi-constraint decisions）
    - 组合型任务（Compositional tasks）
    - 多跳问答（Multi-hop QA）
  - **通常没有帮助的场景**：
    - 缺失背景知识（Missing knowledge —— 此时需要检索 RAG，光靠推理无法无中生有）
    - 任务本身模糊不清（Ambiguous tasks —— 此时需要澄清需求规范）
    - 更适合用外部工具解决的任务（Tasks better solved by tools）

### Slide 27–34：六大核心推理与验证范式（Core Reasoning & Verification Paradigms）

1. **思维链（Chain-of-Thought, CoT, Slides 27–28）**：
   - *Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*。
   - 通过诱导模型输出中间推理步骤（例如添加 `"let's think step-by-step"`）来显著提升多步推理表现。
   - **CoT 类型（Slide 28）**：
     - **零样本（Zero-shot）**：仅提供任务描述，不给任何示例。
     - **少样本（Few-shot）**：提供任务描述并附带若干含推理过程的示例。
     - **8 样本（8-shot）**：提供任务描述并附带恰好 8 个示例（经典的 GSM8K 基准设置）。
   - **适用时机**：多步数学、符号操作、组合推理、子问题规划。

2. **问题分解（Decomposition, Slide 29）**：
   - **通用脚手架（Generic scaffold）**：
     ```text
     List the constraints（列出所有约束条件）
     Break into 3–6 substeps（拆解为 3–6 个子步骤）
     Solve each substep（逐个解决每个子步骤）
     Assemble final answer in the required format（按要求格式组装最终答案）
     ```
   - **“从易到难”分解法（"Least-to-most" style decomposition）**：先从简单的子问题开始回答，再将简单子问题的答案作为已知条件（givens）用于解决更难的步骤。

3. **自洽性解码（Self-Consistency, Slide 30）**：
   - *Wang et al. (2022), "Self-Consistency Improves Chain of Thought Reasoning in Language Models"*（对应作业 2 第 2.4 节）。
   - 利用多次采样选择最高似然（最一致）的答案：
     1. 将温度（temperature）设为较高值 $\rightarrow$ 增加生成路径的多样性（diversity）。
     2. 让 LLM 独立生成 $N$ 条推理路径与解（$N$ independent solutions）。
     3. 选择**多数票（majority vote）**或“最一致（most consistent）”的生成结果。

4. **思维树（Tree of Thoughts, ToT, Slide 31）**：
   - **通用脚手架（Generic scaffold）**：
     ```text
     Define "thought" unit (partial solution / subplan)（定义“思维”单元：局部解或子计划）
     Generate k candidates（在当前节点生成 k 个候选思维分支）
     Score candidates (model-based evaluation, verifier)（评估候选分支得分：基于模型或验证器）
     Expand best nodes until depth/timeout（扩展最优节点直至达到最大深度或超时）
     Return best leaf + justification（返回最优叶节点及其理由）
     ```
   - 将线性 CoT 泛化为在“思维分支树”上的搜索，支持探索（exploration）与前瞻（lookahead），显著提升需要搜索回溯的复杂问题表现。

5. **程序辅助语言模型（Program-Aided Language Models, PAL, Slide 32）**：
   - 将执行正确性卸载给外部运行时环境（如 Python 解释器）。
   - 减少最终计算中的算术失误或逻辑笔误，非常适合需要精确数学推理的问题。
   - **示例**：
     - 问题：*Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?*
     - PAL 输出：将自然语言穿插转换为可执行代码 `tennis_balls = 5`、`bought_balls = 2 * 3`、`answer = tennis_balls + bought_balls` 交由 Python 计算。

6. **自我反思（Reflexion, Slide 33）**：
   - Reflexion 将环境或评估器的反馈（feedback）转化为存储在记忆中的文本化**自我反思（self-reflections）**。
   - 这些反思会在后续尝试中被重复利用 $\rightarrow$ **无需微调权重即可持续提升表现（Improving performance without fine-tuning）**。
   - **通用脚手架**：`Attempt（尝试） → Evaluate（评估） → Write reflection（撰写反思） → Next attempt conditioned on reflection（基于反思进行下一次尝试）`。

7. **自我验证（Self-Verification, Slide 34）**：
   让 LLM 在继续下一步之前先“验证”自己的答案：
   - **自洽性（Self-consistency）**：针对单答案任务的多数投票。
   - **检查器模型（Checker model）**：批评指正（critique）+ 条件重生成（conditioned regeneration，这正是本周 `autoprompting.py` 实验的核心思想）。
   - **基于工具的验证（Tool-based verification）**：用解释器计算、检索引用文献（citations）、交叉核对约束。
   - **不变量检查（Invariant checks）**：Schema 合法性校验（如是否为合法 JSON，对应 HW2 Task 2.5 `validate`）、策略约束检查、数值合理性检查。

### Slide 35：推理失败模式（Reasoning Failure Modes）
- **遗漏约束（Constraint drop）**：LLM 忽略了提示词中的某条 `must`（必须）或 `never`（绝不）规则。
- **过早下结论（Premature closure）**：在考虑完所有约束条件之前就匆忙给出答案。
- **虚假精确（Spurious precision）**：对输入中缺失的数值进行凭空捏造（幻觉）。
- **长上下文不鲁棒 / “迷失在中间”（Non-robust in long context / Lost in the Middle）**：当关键事实没有出现在长提示词的开头或结尾，而是位于中间位置时，即使是长上下文模型，其表现也会显著退化。

### Slide 36–44：推理模型工作原理与 Google AI Studio 配置
- **为什么需要推理模型（Slide 36）**：针对复杂推理问题手写提示工程非常困难，这催生了**推理模型（Reasoning Models）**——自动生成内部中间推理链，再筛选并打磨最终答案。
- **标准 LLM 推理 vs. 推理型 LLM 推理（Slides 37–39）**：
  - **标准 LLM 推理（Slide 37）**：分词（Tokenize） $\rightarrow$ 前向传播（Forward pass） $\rightarrow$ 获取词表 Logits $\rightarrow$ Softmax 转概率分布 $\rightarrow$ 采样输出词元 $\rightarrow$ 追加到序列并循环。
  - **推理型 LLM 推理（Slide 38–39）**：流程相同，但在输出最终回答前，模型会先生成**“思考词元（thinking tokens）”**——包含逐步思考、问题分解、自我验证等结构化中间推理轨迹；这些思考词元不包含在最终呈现给用户的答案正文中（如 Slide 39 展示的 ChatGPT o3 右侧 `Thinking` 面板）。
- **Google AI Studio 与免费配额说明（Slides 40–44）**：
  - 访问 `https://aistudio.google.com/`（使用个人邮箱），创建 API Key，运行 `pip install -U google-genai`。
  - 注意免费层速率限制（Rate Limit）：`gemini-3.5-flash-lite` 每天提供 **500 次免费请求（500 RPD，15 RPM）**，因此请尽早开始完成 HW 2 并善用本地磁盘缓存！

---

## 3. 第三部分：记忆管理（Memory Management, Slides 45–56）

### Slide 46–48：什么是 LLM 的“记忆”？
- **核心定义（Slide 46）**：
  > **“‘Memory’ for an LLM means context selection and presentation.”（对 LLM 而言，“记忆”的本质是上下文的选择与呈现。）**
  - LLM 本身是无状态的；要让它“记住”先前查询中的信息，开发者必须有策略地管理并筛选历史交互中的相关信息，以维持多轮对话的连贯性。
- **LLM 应用中的三层记忆结构（Slides 47–48，以旅行社智能体 Travel Agent 为例）**：
  1. **短期记忆（Short-term memory，即当前对话）**：与用户的当前对话流（存储在上下文窗口 Context Window 中）。
  2. **口袋大小的结构化“事实”（Pocket-sized "facts"）**：美元汇率、天气信息、航班价格/时刻表、或对话中提取出的关键事实（存储在关系型数据库或向量事实库中）。
  3. **长跨度记忆（Long-span memory，跨会话历史）**：用户的饮食禁忌（如素食）、喜好与厌恶、偏好的回复风格等（存储在 NoSQL 数据库或长期画像存储中）。

### Slide 49–52：三种经典对话记忆管理策略对比
（对应作业 2 第 4 部分 `HW2/part4_memory.py`！）

| 策略名称 | 核心机制 | 优点（Pros） | 缺点（Cons） |
| :--- | :--- | :--- | :--- |
| **1. 顺序全量链（Sequential chain, Slide 49）** | 将每一轮新对话原封不动追加到历史末尾，把整条完整对话链作为输入 | 实现最简单，不会丢失任何早期对话细节 | 极快触达上下文窗口上限（Context limits）与高昂 Token 成本，仅适用于极短会话 |
| **2. 滑动窗口（Sliding window / Buffer, Slide 50）** | 仅保留最近 $N$ 轮对话（逐字保留），直接丢弃更早的对话轮次 | 实现简单，上下文长度恒定，绝不会超出窗口限制 | **会永久丢失早期轮次中的重要信息**（例如用户在第 1 轮提供的预订号 `88431`） |
| **3. 摘要记忆与上下文压缩（Summary-based memory / Compaction, Slides 51–52）** | 定期（或当旧轮次滑出窗口时）调用 LLM 将旧对话轮次浓缩为一段简短摘要（Summary） | 在严格受控的 Token 预算下维持较长跨度的全局上下文 | 需要反复迭代调整摘要提示词；**高倍压缩必然丢失细粒度事实**（如精确错误码、配角姓名等） |

### Slide 53–55：智能体记忆的五种工程存储介质（Agentic Memory in Practice）
1. **上下文窗口（Context window）**：LLM 直接处理当前活跃的即时对话历史。
2. **内存缓存（In-Memory Cache，如 Redis）**：亚毫秒级（Sub-millisecond）读取活跃会话状态。
3. **关系型数据库（Relational Database，如 PostgreSQL）**：存储结构化数据（如聊天记录、用户账户、账单记录、汇率/航班表）。
4. **NoSQL 数据库（NoSQL Database，如 MongoDB/Firestore）**：存储高度动态或非结构化的数据（如智能体执行轨迹日志、用户长期偏好画像）。
5. **向量数据库 + 词嵌入（Vector Database + Embeddings）**：基于语义相似度而非精确关键词执行语义搜索（Semantic search），按需拉取与当前问题最相关的历史事实（即 RAG / `FactStore`）。

---

## 4. 第四部分：课堂讨论题解答（Slide 56 Discussion）

> **题目要求（Slide 56）**：
> - **Step 1**：构思一个需要多轮交互、约束条件随对话不断演变、依赖记住先前决策、且一旦遗忘早期信息就会失败的 LLM 任务。
> - **Step 2**：为该任务设计一个记忆存储系统，包含：(1) 从对话中提取相关信息的方法；(2) 将总长度严格控制在 **500 tokens 以内**的存储 Schema。

### 参考解答（Reference Solution）

#### Step 1：任务场景定义 —— 多轮临床分诊与处方核查智能体（Multi-Turn Clinical Triage & Medication Reconciliation Agent）
- **为什么满足全部四个条件**：
  1. **多轮交互（Multiple turns）**：患者在 15–25 轮对话中逐步描述症状、既往病史、当前正在服用的药物以及药物过敏史。
  2. **约束随对话动态演变（Evolving constraints）**：第 2 轮患者提到“对青霉素过敏”，第 8 轮提到“有轻度肾功能不全（eGFR 42）”，第 14 轮又补充“昨晚开始服用布洛芬但胃痛”——每一轮新信息都会收紧或改变后续用药建议的安全约束。
  3. **依赖先前决策（Depends on remembering prior decisions）**：第 6 轮已排除细菌性咽炎并建议停用某非处方药，后续轮次不能自相矛盾地再次推荐同类药物。
  4. **遗忘早期信息会导致严重失败（Fails if earlier information is forgotten）**：如果滑动窗口在第 20 轮丢弃了第 2 轮的“青霉素过敏”或“华法林用药史”，智能体可能给出致命的药物相互作用建议。

#### Step 2：记忆提取方法与 `< 500 Tokens` 紧凑存储 Schema 设计
1. **信息提取方法（Extraction Method）**：
   - 采用**“增量式结构化状态更新 + 语义事实抽取”双通道机制**：
   - 每当一轮对话 `(user_turn, assistant_turn)` 结束时，触发一个轻量级后台提取调用（Temperature = 0.0），仅提取**新增或变更的临床约束、患者硬性属性与已达成的诊疗决定**，并以 JSON Patch 形式合并到固定槽位的状态对象中；同时将滑出最近 4 轮缓冲区的旧对话压缩为 $\le 60$ 词的病情进展摘要。
2. **严格控制在 500 Tokens 以内的三层存储 Schema（约 280–350 Tokens）**：
   ```json
   {
     "patient_profile": {
       "age": 54,
       "sex": "F",
       "renal_hepatic_flags": ["CKD Stage 3b (eGFR 42)"]
     },
     "hard_constraints": {
       "allergies": ["Penicillin (anaphylaxis)", "Sulfa drugs (rash)"],
       "active_medications": ["Warfarin 5mg QD", "Lisinopril 10mg QD"],
       "contraindicated_classes": ["NSAIDs", "Beta-lactams"]
     },
     "prior_decisions": [
       "Turn 6: Ruled out strep throat (Centor score 1); advised against antibiotics.",
       "Turn 14: Advised immediate discontinuation of Ibuprofen due to Warfarin + CKD."
     ],
     "rolling_summary": "54F presenting with 3-day low-grade fever and viral pharyngitis. Experiencing mild epigastric pain after self-medicating with ibuprofen. Currently stable, seeking safe analgesic alternative.",
     "recent_buffer": [
       {"role": "user", "content": "What can I take for my headache and sore throat instead of ibuprofen?"},
       {"role": "assistant", "content": "Given your Warfarin therapy and kidney function, acetaminophen (up to 2,000mg/day) is the safest option."}
     ]
   }
   ```
   - **Token 预算核算**：
     - `patient_profile` + `hard_constraints`（不可丢弃的安全红线槽位）：$\le 100$ tokens
     - `prior_decisions`（最多保留最近 5 条关键决策）：$\le 80$ tokens
     - `rolling_summary`（上限 60 词的滚动摘要）：$\le 80$ tokens
     - `recent_buffer`（最近 2–4 轮原文）：$\le 150$ tokens
     - **总计**：$\approx 320\text{–}410\text{ tokens} < 500\text{ tokens}$，既保证了硬性安全约束 0 丢失，又保持了极高的上下文效率。

---

## 5. 第五部分：课堂编程实验详解（Autoprompting & Gemini API）

第 4 周课堂配备了两个动手实验脚本：
1. [`labs/gemini.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/gemini.py)（中文版 [`labs/gemini_zh.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/gemini_zh.py)）：验证 `google-genai` SDK 与环境变量配置。
2. [`labs/autoprompting.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/autoprompting.py)（中文版 [`labs/autoprompting_zh.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/02-columbia-designing-agents/labs/autoprompting_zh.py)）：构建一个自动提示优化闭环（Autoprompting Loop）：
   - **Task 0 (`GENERATOR_PROMPT`)**：从最简单的种子提示词 `"Write a short poem about capitalism."` 出发。
   - **Task 1 (`ERROR_TAXONOMY`)**：定义 6 类可观测、可复现的诗歌缺陷分类标准（`cliche` 陈词滥调、`forced rhyme` 强行押韵、`sing-song meter` 顺口溜式呆板节拍、`stated feeling` 直接宣泄情绪而非通过意象展现、`moral ending` 结尾空洞说教、`padding` 同义反复凑字数）。
   - **Task 2 (`judge`)**：实现 **LLM-as-a-Judge**（Temperature = 0.0），自动诊断当前生成诗歌包含哪些缺陷。
   - **Task 3 (`rewrite`)**：实现 **Meta-Prompt Rewriter**，将原提示词、生成的诗歌及诊断出的缺陷描述一并交给模型，生成针对性规避这些缺陷的新提示词。
   - **Task 4 (`autoprompt`)**：执行 3 轮迭代重写（共 4 轮评估），错误数量从第 0 轮的 **3 个**（`cliche, sing-song meter, moral ending`）逐步下降至第 1 轮的 **2 个**、第 2 轮的 **1 个**，并在第 3 轮收敛至 **0 个错误（`none`）**。
