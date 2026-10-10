# Course 09: SJTU 上海交通大学《动手学大模型智能体》(Hands-on AI Agent)

**University**: Shanghai Jiao Tong University (上海交通大学)  
**Instructors / Authors**: Prof. Weinan Zhang (张伟楠) et al. / Boyu-AI (伯禹教育)  
**Official Repository**: [https://github.com/boyu-ai/Hands-on-AA](https://github.com/boyu-ai/Hands-on-AA)  

---

## 1. Course Overview & 15-Chapter Architecture

This repository contains complete, runnable Python implementations from scratch and detailed Chinese study notes ([`SJTU_Hands_On_AA_Notes_zh.md`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/SJTU_Hands_On_AA_Notes_zh.md)) for all **4 Parts (15 Chapters)** of SJTU's *Hands-on AI Agent* (`动手学大模型智能体`).

| Part | Chapter Module | Core Concepts & Implemented Components |
| :--- | :--- | :--- |
| **Part 1: Foundations**<br>(`part1_foundations/`) | [`ch01_agent_intro.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch01_agent_intro.py) | POMDP perception-thought-action loop, `SmartHomeEnv`, `StatefulLLMAgentWrapper` |
| | [`ch02_generative_lm.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch02_generative_lm.py) | Causal self-attention from scratch, `KVCache` incremental decoding equivalence, Temperature / Top-$k$ / Top-$p$ sampling |
| | [`ch03_prompt_engineering.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch03_prompt_engineering.py) | System + Few-Shot + CoT prompt templates, `SelfConsistencyVoter` majority voting & entropy, `StructuredJSONParser` with syntax repair |
| | [`ch04_agent_evaluation.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part1_foundations/ch04_agent_evaluation.py) | Unbiased `pass@k` estimator, `TrajectoryEvaluator` (Precision, Recall, F1, LCS Milestone Progress), `AgentCostTracker` |
| **Part 2: Architectures**<br>(`part2_architectures/`) | [`ch05_react_and_plan_solve.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch05_react_and_plan_solve.py) | `ReActAgent` interleaved loop vs. `PlanAndSolveAgent` (ReWOO symbolic variable binding `#E1`, `#E2`) |
| | [`ch06_memory_and_rag.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch06_memory_and_rag.py) | `WorkingMemoryBuffer`, `GenerativeEpisodicMemory` (Recency + Importance + Relevance), `HybridRAGEngine` (BM25 + Dense Vector RRF) |
| | [`ch07_tool_calling.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch07_tool_calling.py) | Automatic function signature -> JSON Schema `ToolRegistry`, runtime type validation, structured error recovery hints |
| | [`ch08_tree_search_tot_mcts.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part2_architectures/ch08_tree_search_tot_mcts.py) | `ToTBeamSearch` (Tree-of-Thoughts BFS) and `MCTSReasoningAgent` (UCT selection, expansion, simulation, backpropagation on Game of 24) |
| **Part 3: Post-Training**<br>(`part3_training/`) | [`ch09_instruction_sft.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part3_training/ch09_instruction_sft.py) | `AgentTrajectoryCollator` multi-turn loss masking (`IGNORE_INDEX = -100` on user/tool spans) & `compute_masked_sft_loss` |
| | [`ch10_lora_and_quantization.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part3_training/ch10_lora_and_quantization.py) | `LoRALinear` ($W_0 + \frac{\alpha}{r}BA$, analytical gradients, lossless weight merge) + `quantize_int8_absmax` & `quantize_int4_groupwise` |
| | [`ch11_rl_dpo_grpo.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part3_training/ch11_rl_dpo_grpo.py) | `compute_dpo_loss` (implicit reward margins) & `compute_grpo_objective` (critic-free group-relative advantages + unbiased KL penalty) |
| **Part 4: Frontiers**<br>(`part4_frontiers/`) | [`ch12_multimodal_gui_agent.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch12_multimodal_gui_agent.py) | Normalized `[0, 1000]` `BoundingBox` to pixel coordinate mapping, IoU metric, Set-of-Mark `MockGUIEnvironment` (`CLICK`, `TYPE`, `SCROLL`, `FINISH`) |
| | [`ch13_multi_agent_systems.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch13_multi_agent_systems.py) | `SOPSoftwareCompanyPipeline` (MetaGPT-style PM -> Architect -> Engineer -> Sandbox QA) & `MultiAgentDebateArena` |
| | [`ch14_agent_safety.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch14_agent_safety.py) | `PromptInjectionDetector` (direct/indirect), `PIIRedactionFilter` (API keys, emails, phones, cards), `ToolPermissionPolicyEngine` (RBAC + HITL gate) |
| | [`ch15_agent_protocols.py`](file:///Users/yixin0909zhang/Downloads/GDG/ai-agents-courses-homework/09-sjtu-hands-on-aa/part4_frontiers/ch15_agent_protocols.py) | `MCPServer` (`initialize`, `tools/list`, `tools/call`, `resources/read`) & `A2ARouter` (`AgentCard` discovery + `tasks/send` JSON-RPC 2.0) |

---

## 2. Running the Test Suite & Chapter Demos

Run the complete `pytest` verification suite across all 15 chapters:

```bash
pytest -v test_all_15_chapters.py
```

Each chapter module can also be executed directly as a standalone script:

```bash
python3 part1_foundations/ch01_agent_intro.py
python3 part2_architectures/ch08_tree_search_tot_mcts.py
python3 part3_training/ch11_rl_dpo_grpo.py
python3 part4_frontiers/ch15_agent_protocols.py
```
