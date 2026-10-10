# Stanford CS329A: Self-Improving AI Agents — Complete Homework & Lab Solutions

**Course:** Stanford CS329A — *Self-Improving AI Agents* (Winter 2025, Instructors: Aakanksha Chowdhery & Azalia Mirhoseini, `https://cs329a.stanford.edu/`)  
**Verification Status:** 13/13 unit tests passing (`pytest hw1/ hw2/ hw3/ -v`)

---

## Repository Structure

| Module | Implementation & Tests | Technical Report | Core Algorithms & Topics Covered |
| :--- | :--- | :--- | :--- |
| **HW1: Test-Time Compute Scaling & Self-Debugging** | [`hw1/test_time_scaling.py`](./hw1/test_time_scaling.py)<br>[`hw1/test_hw1.py`](./hw1/test_hw1.py) | [`hw1/HW1_Report.md`](./hw1/HW1_Report.md) | Unbiased `pass@k` estimator (Chen et al., 2021), Self-Consistency Majority Voting, ORM vs. Step-level PRM `Best-of-N` & Verifier-Weighted Voting, Execution-Feedback Self-Debugging loop |
| **HW2: Tree Search Planning & RL Post-Training** | [`hw2/lats_and_grpo.py`](./hw2/lats_and_grpo.py)<br>[`hw2/test_hw2.py`](./hw2/test_hw2.py) | [`hw2/HW2_Report.md`](./hw2/HW2_Report.md) | Language Agent Tree Search (`LATS` MCTS + UCT + Verbal Reflection), `ADaPT` (As-Needed Decomposition & Planning), `STaR` rationale filtering/rationalization, `GRPO` group-normalized advantage & Schulman K3 KL loss |
| **HW3: Agentic Search & Kernel Optimization** | [`hw3/search_and_kernel_agent.py`](./hw3/search_and_kernel_agent.py)<br>[`hw3/test_hw3.py`](./hw3/test_hw3.py) | [`hw3/HW3_Report.md`](./hw3/HW3_Report.md) | `SearchO1Agent` (`<|begin_search_query|>` + Reason-in-Documents compression), `KernelOptimizationAgent` (KernelBench numerical verification + runtime latency profiling) |

---

## Quickstart: Running Benchmarks & Unit Tests

```bash
# Run all unit tests across HW1, HW2, and HW3
python3 -m pytest hw1/ hw2/ hw3/ -v

# Execute individual homework benchmark suites
python3 hw1/test_time_scaling.py
python3 hw2/lats_and_grpo.py
python3 hw3/search_and_kernel_agent.py
```
