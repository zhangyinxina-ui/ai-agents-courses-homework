# 上海交通大学《动手学大模型智能体》(Hands-on AI Agent) 全15章深度讲义与代码精解

**课程作者**：上海交通大学 张伟楠（Weinan Zhang）教授团队 / 伯禹教育 (`boyu-ai/Hands-on-AA`)  
**官方代码仓库**：[https://github.com/boyu-ai/Hands-on-AA](https://github.com/boyu-ai/Hands-on-AA)  
**本目录实现**：完整覆盖全书 **4 大篇章、15 个核心章节** 的可运行从零实现（Pure Python + NumPy）与自动化单元测试套件（[`test_all_15_chapters.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/test_all_15_chapters.py)）。

---

## 全书体系架构总览

```
+========================================================================================+
|                   上海交通大学《动手学大模型智能体》四篇十五章全景图                   |
+========================================================================================+
| 第一篇：智能体基础篇 (Part 1: Foundations, Ch01–Ch04)                                   |
|   ├── Ch01 初识智能体：POMDP 感知-思考-行动闭环、反应式与状态目标导向智能体             |
|   ├── Ch02 生成式语言模型：因果自注意力、KV Cache 增量解码等价性证明、Top-k/Top-p 采样 |
|   ├── Ch03 提示工程与推理激发：Few-Shot、CoT 思维链、Self-Consistency 自洽投票、JSON解析|
|   └── Ch04 智能体评测方法：无偏 pass@k 估计器、工具轨迹 Precision/Recall/LCS、成本追踪  |
+----------------------------------------------------------------------------------------+
| 第二篇：核心架构篇 (Part 2: Core Architectures, Ch05–Ch08)                             |
|   ├── Ch05 推理与规划范式：ReAct 交错闭环 vs. Plan-and-Solve / ReWOO 变量解耦绑定      |
|   ├── Ch06 记忆与检索增强：滑动窗口摘要记忆、Generative Agents 三维记忆评分、混合RRF检索|
|   ├── Ch07 工具调用与执行：Python函数自动生成 JSON Schema、参数类型校验与自纠错恢复提示|
|   └── Ch08 树搜索深层推理：Tree-of-Thoughts (ToT Beam/DFS) 与 MCTS (UCT 四阶段搜索)    |
+----------------------------------------------------------------------------------------+
| 第三篇：后训练与微调篇 (Part 3: Post-Training & RL, Ch09–Ch11)                         |
|   ├── Ch09 指令微调与轨迹SFT：多轮交互轨迹掩码 (IGNORE_INDEX=-100) 与因果交叉熵梯度推导|
|   ├── Ch10 参数高效微调与量化：LoRA 低秩分解/零初始化/权重合并、INT8 AbsMax与INT4分组量化|
|   └── Ch11 智能体强化学习：DPO 直接偏好优化隐式奖励边际、GRPO 组相对优势与无偏KL惩罚项 |
+----------------------------------------------------------------------------------------+
| 第四篇：前沿范式与生态篇 (Part 4: Frontiers & Protocols, Ch12–Ch15)                    |
|   ├── Ch12 多模态与GUI智能体：Set-of-Mark (SoM) 视觉定位、[0,1000]坐标映射与GUI动作空间|
|   ├── Ch13 多智能体协作系统：MetaGPT 标准作业程序 (SOP) 角色流水线与多智能体辩论共识    |
|   ├── Ch14 智能体安全与对齐：直接/间接提示注入检测、PII隐私脱敏过滤器、RBAC最小权限引擎|
|   └── Ch15 智能体通信协议：MCP (Model Context Protocol) 与 A2A 跨智能体 JSON-RPC 路由  |
+========================================================================================+
```

---

## 第一篇：智能体基础篇 (`part1_foundations/`)

### 第 1 章：初识智能体（[`ch01_agent_intro.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch01_agent_intro.py)）
1. **数学建模（POMDP）**：
   智能体与外部环境的交互可建模为部分可观测马尔可夫决策过程 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R})$。在每一步 $t$，环境发出观测 $o_t \in \mathcal{O}$，智能体更新内部历史记忆 $h_t = (o_0, a_0, \dots, o_t)$，生成显式思考链 $z_t \sim \pi_\theta(z_t \mid h_t)$ 并执行动作 $a_t \sim \pi_\theta(a_t \mid h_t, z_t)$。
2. **代码实现要点**：
   - 实现了多房间智能家居环境 `SmartHomeEnv`（包含房间移动 `MOVE`、清洁 `CLEAN`、温控调节 `SET_TEMP`、灯光控制 `SWITCH_LIGHTS` 及终止动作 `FINISH`）。
   - 实现了 `StatefulLLMAgentWrapper`，记录完整 `(step, observation, thought, action)` 轨迹并验证目标达成率。

### 第 2 章：生成式语言模型基础（[`ch02_generative_lm.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch02_generative_lm.py)）
1. **因果自注意力与 KV Cache 增量推理**：
   - 全序列计算中，$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{causal}}\right)V$。
   - 在自回归生成第 $t$ 个 Token 时，由于因果掩码 $M_{\text{causal}}$ 的下三角性质，历史位置 $1 \dots t-1$ 的键向量 $K_{1:t-1}$ 和值向量 $V_{1:t-1}$ 保持不变。通过 `KVCache` 缓存历史键值，单步解码仅需计算当前查询向量 $q_t \in \mathbb{R}^{1 \times d_k}$ 与拼接后 $K_{1:t}$ 的点积，将单步计算复杂度从 $\mathcal{O}(t^2 d)$ 降至 $\mathcal{O}(t d)$。
2. **解码采样策略**：
   - **Temperature 温度缩放**：$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$，$\tau \to 0$ 退化为贪心解码。
   - **Top-$k$ 采样**：仅保留概率最高的 $k$ 个候选词并重新归一化。
   - **Top-$p$（Nucleus 核采样）**：按概率降序排列后，截取最小集合 $V^{(p)}$ 使得累积概率 $\sum_{i \in V^{(p)}} p_i \ge p$。
3. **代码实现验证**：`ch02_generative_lm.py` 验证了带 `KVCache` 的增量解码与全序列前向传播的最大 Logit 误差小于 $10^{-12}$。

### 第 3 章：提示工程与推理激发（[`ch03_prompt_engineering.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch03_prompt_engineering.py)）
1. **思维链（CoT）与自洽性（Self-Consistency）**：
   - 通过温度采样生成 $K$ 条多元推理路径 $\{ (z^{(k)}, a^{(k)}) \}_{k=1}^K$，并对最终答案执行多数投票：
     $$\hat{a} = \arg\max_{a} \sum_{k=1}^K \mathbb{I}(a^{(k)} = a)$$
   - 同时计算投票分布的信息熵 $H = -\sum_{a} P(a) \log_2 P(a)$ 作为智能体输出的置信度不确定性指标。
2. **结构化 JSON 解析与语法修复**：
   `StructuredJSONParser` 自动剥离 Markdown 代码块、修复尾随逗号（Trailing Comma），并严格校核必选字段与数据类型。

### 第 4 章：智能体评测方法与基准（[`ch04_agent_evaluation.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch04_agent_evaluation.py)）
1. **无偏 `pass@k` 估计量（Codex / HumanEval 标准公式）**：
   生成 $n \ge k$ 个候选样本，其中 $c$ 个通过测试，直接计算 $1 - (1 - \hat{p})^k$ 会产生方差偏差，无偏估计公式为：
   $$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} = 1 - \prod_{i=0}^{k-1} \frac{n - c - i}{n - i}$$
2. **工具调用轨迹评测与里程碑进度（Milestone Progress）**：
   `TrajectoryEvaluator` 同时评估工具调用集合的 Precision、Recall、F1，以及基于最长公共子序列（LCS）的时序里程碑达成率 $\text{LCS}(\tau_{\text{pred}}, \tau_{\text{gold}}) / |\tau_{\text{gold}}|$。

---

## 第二篇：核心架构篇 (`part2_architectures/`)

### 第 5 章：推理与规划范式 — ReAct 与 Plan-and-Solve（[`ch05_react_and_plan_solve.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch05_react_and_plan_solve.py)）
1. **ReAct（Reason + Act）交错范式**：
   每一步均采用 `Thought -> Action -> Observation` 闭环，适应动态环境变化，但每步都需要将累积历史重新送入 LLM，调用次数与行数成正比。
2. **Plan-and-Solve / ReWOO（Reasoning Without Observation）解耦范式**：
   Planner 一次性生成带符号变量占位符（`#E1`, `#E2`, `#E3`）的有向无环图（DAG）执行计划；Worker 按拓扑序解析变量并执行工具，将多步问答的 LLM 调用次数从 $N$ 次压缩至 **2 次**。

### 第 6 章：智能体记忆与知识检索增强（[`ch06_memory_and_rag.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch06_memory_and_rag.py)）
1. **斯坦福 Generative Agents 三维记忆检索评分**：
   综合考量记忆条目的**近因性（Recency）**、**重要性（Importance）**与**语义相关性（Relevance）**：
   $$S(m, q) = w_{\text{rec}} \cdot \gamma^{\Delta t} + w_{\text{imp}} \cdot \frac{\text{Imp}(m)}{10} + w_{\text{rel}} \cdot \frac{\cos(\mathbf{e}_m, \mathbf{e}_q) + 1}{2}$$
2. **混合检索与倒数排序融合（Hybrid RAG with RRF）**：
   结合 BM25 稀疏词法匹配与稠密向量余弦相似度，通过 **Reciprocal Rank Fusion (RRF)** 融合排名：
   $$\text{RRF}(d) = \frac{1}{k + \text{rank}_{\text{BM25}}(d)} + \frac{1}{k + \text{rank}_{\text{dense}}(d)}, \quad (k = 60)$$

### 第 7 章：工具调用与外部环境交互（[`ch07_tool_calling.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch07_tool_calling.py)）
- 利用 Python `inspect.signature` 与 `get_type_hints` 自动将任意 Python 函数反射为标准 OpenAI / MCP JSON Schema。
- 当大模型传入错误类型的参数（例如将 `float` 误传为 `str` 或将 `int` 误传为 `bool`）时，`ToolRegistry.dispatch` 捕获异常并生成结构化 `recovery_hint`，指导智能体在下一轮自动修正参数调用。

### 第 8 章：树搜索与深层推理 — ToT 与 MCTS（[`ch08_tree_search_tot_mcts.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch08_tree_search_tot_mcts.py)）
- **思维树（Tree-of-Thoughts, ToT）**：维护宽度为 $b$ 的候选中间状态束（Beam Search），在算术 24 点问题上系统展开算子分支。
- **蒙特卡洛树搜索（MCTS）**：实现完整四阶段循环——**选择（Selection）**、**扩展（Expansion）**、**模拟（Simulation）**、**反向传播（Backpropagation）**，其中节点选择遵循 UCT 公式：
  $$\text{UCT}(s, a) = \frac{W(s, a)}{N(s, a)} + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

---

## 第三篇：后训练与微调篇 (`part3_training/`)

### 第 9 章：指令微调与智能体轨迹 SFT（[`ch09_instruction_sft.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part3_training/ch09_instruction_sft.py)）
- **多轮轨迹损失掩码（Loss Masking）**：在智能体多轮轨迹 $\tau = (\text{System}, \text{User}, \text{Assistant}_1, \text{ToolObs}_1, \text{Assistant}_2)$ 的监督微调中，只能对 `assistant` 角色生成的 Token 计算交叉熵损失，而必须将 `system`、`user` 和 `tool` 环境返回的观测 Token 标签设为 `IGNORE_INDEX = -100`：
  $$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\sum_{t} \mathbb{I}(y_t \neq -100)} \sum_{t: y_t \neq -100} \log \pi_\theta(y_t \mid y_{<t})$$
- 代码严格验证了被掩码的 `tool`/`user` 位置梯度范数恒等于 `0.0`。

### 第 10 章：参数高效微调 LoRA 与模型量化（[`ch10_lora_and_quantization.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part3_training/ch10_lora_and_quantization.py)）
1. **LoRA（Low-Rank Adaptation）**：
   冻结预训练权重 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，注入低秩矩阵 $A \in \mathbb{R}^{r \times d_{\text{in}}}$（高斯初始化）与 $B \in \mathbb{R}^{d_{\text{out}} \times r}$（全零初始化，确保训练初始时刻 $\Delta W = 0$）：
   $$h = x W_0^\top + \frac{\alpha}{r} x A^\top B^\top$$
   推理部署时可无损合并为 $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$，实现**零额外推理延迟**。
2. **INT8 对称量化与 INT4 分组非对称量化**：
   实现了 AbsMax INT8 量化与 GPTQ/AWQ 风格的分组（`group_size=8`）零点缩放 INT4 量化，并对比验证了量化重构均方误差（MSE）。

### 第 11 章：智能体强化学习 — DPO 与 GRPO（[`ch11_rl_dpo_grpo.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part3_training/ch11_rl_dpo_grpo.py)）
1. **直接偏好优化（DPO）**：
   利用闭式映射 $r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$ 绕过显式奖励模型建模：
   $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$
2. **群组相对策略优化（GRPO，DeepSeekMath / DeepSeek-R1 核心算法）**：
   针对同一提示词采样 $G$ 条智能体轨迹 $\{o_1, \dots, o_G\}$，使用组内标准化奖励直接作为优势函数 $A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}$，彻底省去与策略模型同规模的 Value Critic 网络显存开销，并结合无偏非负 KL 散度估计器：
   $$\mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1$$

---

## 第四篇：前沿范式与生态篇 (`part4_frontiers/`)

### 第 12 章：多模态与图形界面 GUI 智能体（[`ch12_multimodal_gui_agent.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch12_multimodal_gui_agent.py)）
- 实现了多模态大模型常用的 `[0, 1000]` 归一化边界框 `[ymin, xmin, ymax, xmax]` 到物理屏幕像素 `(1920x1080)` 的坐标转换与交并比（IoU）计算。
- 构建了带 Set-of-Mark (SoM) 元素的浏览器环境，演示智能体通过 `CLICK`、`TYPE`、`SCROLL`、`FINISH` 动作序列自动完成机票搜索表单填写与滚动提交。

### 第 13 章：多智能体协作系统（[`ch13_multi_agent_systems.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch13_multi_agent_systems.py)）
- **MetaGPT 标准作业程序（SOP）软件开发流水线**：产品经理（PRD）$\to$ 架构师（ARCH_SPEC）$\to$ 工程师（SOURCE_CODE）$\to$ 测试工程师（QA_REPORT，在沙箱中动态执行生成代码并断言测试通过）。
- **多智能体辩论（Multi-Agent Debate）**：展示了多个智能体在多轮互评中发现算术推理失误并达成 100% 正确共识的过程。

### 第 14 章：智能体安全、隐私与对齐防护（[`ch14_agent_safety.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch14_agent_safety.py)）
- **三层纵深防御（Defense-in-Depth）**：
  1. `PromptInjectionDetector`：拦截直接越狱指令与检索文档中潜藏的间接提示注入（Indirect Prompt Injection）。
  2. `PIIRedactionFilter`：在上下文落盘或外发前自动脱敏 API 密钥（`sk-...`）、电子邮箱、手机号与银行卡号。
  3. `ToolPermissionPolicyEngine`：执行基于角色的最小权限控制（RBAC）、高危操作人工审批门控（Human-in-the-Loop Gate）以及危险 SQL/Shell 注入参数阻断。

### 第 15 章：智能体通信与互联协议 — MCP 与 A2A（[`ch15_agent_protocols.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch15_agent_protocols.py)）
- **MCP（Model Context Protocol，模型上下文协议）**：实现了完整的 JSON-RPC 2.0 服务端生命周期（`initialize` 协议版本协商、`tools/list` 工具发现、`tools/call` 工具调用、`resources/read` 统一资源读取）。
- **A2A（Agent-to-Agent，跨智能体协作协议）**：实现了基于 `AgentCard` 的技能服务发现（`discover_by_skill`）与 `tasks/send` 任务委派路由，演示了上层协调智能体通过 A2A 发现硬件 PPA 专家智能体、后者内部调用 MCP 工具完成 64 位 ALU 面积与延迟评估的端到端互联闭环。
