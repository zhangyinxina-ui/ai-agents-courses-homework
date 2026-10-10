# Stanford CS329A — Homework 1 Technical Report
## Test-Time Compute Scaling, Step/Outcome Verifiers, and Execution-Feedback Self-Debugging

**Course:** Stanford CS329A — *Self-Improving AI Agents* (Instructors: Aakanksha Chowdhery & Azalia Mirhoseini)  
**Implementation:** [`hw1/test_time_scaling.py`](./test_time_scaling.py) | **Test Suite:** [`hw1/test_hw1.py`](./test_hw1.py)

---

## 1. Theoretical Foundations & Mathematical Formulations

### 1.1 Repeated Sampling and the Unbiased $\text{pass}@k$ Estimator
When scaling test-time compute via parallel sampling, we draw $n \ge k$ independent reasoning trajectories $\{y_1, \dots, y_n\} \sim \pi_\theta(\cdot \mid x)$ at temperature $T > 0$, out of which $c \in \{0, \dots, n\}$ trajectories yield the correct ground-truth solution. Directly computing the empirical success rate on a single subset of size $k$ has high variance, while $1 - (1 - \hat{p})^k$ is a biased estimator. Following Chen et al. (2021), [`estimate_pass_at_k`](./test_time_scaling.py) implements the exact, minimum-variance unbiased estimator:

$$\text{pass}@k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} = 1 - \prod_{i=0}^{k-1} \frac{n - c - i}{n - i}$$

This product formulation avoids large factorial overflow and satisfies exact boundary identities: $\text{pass}@k = 0$ when $c = 0$, and $\text{pass}@k = 1$ whenever $n - c < k$.

### 1.2 Self-Consistency (Majority Voting) vs. Verifier-Guided Selection
Given $N$ sampled trajectories $\{(r_i, a_i)\}_{i=1}^N$ where $r_i = (s_{i,1}, \dots, s_{i,T_i})$ is the multi-step reasoning chain and $a_i = \text{extract}(r_i)$ is the normalized final answer:

1. **Unweighted Majority Voting (Self-Consistency, Wang et al., 2023):** Marginalizes over reasoning paths by selecting the most frequent normalized final answer:
   $$\hat{a}_{\text{maj}} = \arg\max_{a} \sum_{i=1}^N \mathbb{I}[a_i = a]$$
   *Failure Mode:* When multiple distinct arithmetic or conceptual errors converge on the same "attractor" distractor answer (e.g., forgetting a carry or misreading a sign), majority voting fails whenever the distractor mode exceeds the correct answer's frequency.

2. **Outcome Reward Model (ORM) vs. Process Reward Model (PRM) Best-of-$N$ (Lightman et al., 2023):**
   - **ORM (`mode="orm"`):** Assigns a scalar score $V_{\text{ORM}}(x, r_i, a_i) \in (0, 1)$ to the complete solution holistically.
   - **PRM (`mode="prm_min"`):** Scores every intermediate reasoning step $s_{i,t}$ individually via $v(s_{i,t} \mid x, s_{i,<t}) \in (0, 1)$ and aggregates across the trajectory using the bottleneck minimum operator:
     $$V_{\text{PRM-min}}(r_i) = \min_{t \in \{1, \dots, T_i\}} v(s_{i,t} \mid x, s_{i,<t})$$
   - **Standard Best-of-$N$ (BoN):** $\hat{a}_{\text{BoN}} = a_{i^*}$ where $i^* = \arg\max_{i \in \{1,\dots,N\}} V(r_i)$.
   - **Verifier-Weighted Majority Voting (Snell et al., 2024):** Combines path diversity with step-level verification by weighting each vote by its PRM score:
     $$\hat{a}_{\text{PRM-W}} = \arg\max_{a} \sum_{i=1}^N \mathbb{I}[a_i = a] \cdot V_{\text{PRM-min}}(r_i)$$

### 1.3 Execution-Feedback Self-Debugging (Sequential Test-Time Compute)
Whereas Best-of-$N$ scales test-time compute **in parallel**, [`SelfDebuggingAgent`](./test_time_scaling.py) scales compute **sequentially** across $T$ refinement turns (Chen et al., 2023). At turn $t$:
1. Candidate code $c^{(t)}$ is compiled and executed against unit tests $\mathcal{U} = \{u_1, \dots, u_m\}$ inside an isolated namespace via [`execute_python_with_tests`](./test_time_scaling.py).
2. If all assertions pass, execution halts early and returns $c^{(t)}$.
3. Otherwise, the agent captures the exact failing assertion $u_j$, exception class, error message, and stack traceback, synthesizes a diagnostic verbal reflection $\rho^{(t)}$, and generates repaired code $c^{(t+1)} = \pi_{\text{repair}}(x, c^{(t)}, \text{Traceback}^{(t)}, \rho^{(t)})$.

---

## 2. Empirical Scaling Results

### 2.1 Parallel Test-Time Scaling on the CS329A Math Reasoning Suite (`MATH_BENCHMARK`)
We evaluate 6 multi-step arithmetic and combinatorics problems across $N \in \{1, 2, 4, 8, 16, 32\}$ sampled candidates per problem (40 Monte Carlo trials per configuration, base single-sample accuracy $p_0 = 36\%$, seed `2025`):

| Sample Budget $N$ | Unweighted Majority Vote | ORM Best-of-$N$ | PRM (Step-Min) Best-of-$N$ | PRM-Weighted Voting | Oracle $\text{pass}@N$ Upper Bound |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 1$** | 38.33% | 38.33% | 38.33% | 38.33% | 38.33% |
| **$N = 2$** | 50.83% | 62.92% | 62.92% | 62.92% | 62.92% |
| **$N = 4$** | 55.42% | 83.75% | 83.75% | 83.75% | 83.75% |
| **$N = 8$** | 67.50% | 96.67% | 96.67% | 96.67% | 96.67% |
| **$N = 16$** | 78.33% | 99.58% | 99.58% | 99.58% | 99.58% |
| **$N = 32$** | 88.33% | 100.00% | 100.00% | 100.00% | 100.00% |

**Key Empirical Takeaways:**
1. **Compute Efficiency Multiplier ($4\times$–$8\times$ sample reduction):** PRM Best-of-$N$ and PRM-Weighted Voting at $N = 4$ ($83.75\%$) already outperform Unweighted Majority Voting at $N = 16$ ($78.33\%$), achieving a $>4\times$ reduction in test-time sampling compute for the same target accuracy.
2. **Coherent Distractor Vulnerability:** On problems such as `math_01_Widgets` ($18 \times 6 + 24 \times 4 = 204$), two distinct single-step multiplication slips ($98 + 96 = 194$ and $108 + 86 = 194$) concentrate probability mass on the wrong answer `194`. Unweighted Majority Voting requires large $N$ to overcome this bias, whereas `VerifierModel.score_process(..., aggregation="min")` immediately penalizes the invalid step ($v = 0.08$) and selects the verified trajectory ($v = 0.99$) as soon as at least one correct sample appears in the pool.

---

### 2.2 Sequential Test-Time Scaling: Execution-Feedback Self-Debugging (`CODING_BENCHMARK`)
We evaluate [`SelfDebuggingAgent`](./test_time_scaling.py) on 5 buggy algorithmic coding tasks (`sum_to_n`, `min_coins` DP, `is_valid_palindrome`, `binary_search_first`, `merge_intervals`) across turn budgets $T_{\max} \in \{1, 2, 3, 4\}$:

| Max Debugging Turns ($T_{\max}$) | Tasks Solved | Pass Rate (%) | Repaired Bug Classes |
| :---: | :---: | :---: | :--- |
| **Turn 1 (Zero-Shot Pass@1)** | $0 / 5$ | **0.0%** | Initial buggy implementations fail boundary/DP/slice assertions |
| **Turn 2 (+1 Exec Feedback)** | $1 / 5$ | **20.0%** | Repairs off-by-one loop bound (`range(1, n + 1)` in `sum_to_n`) |
| **Turn 3 (+2 Exec Feedbacks)** | $4 / 5$ | **80.0%** | Repairs 2-bug chains in `min_coins` (`inf` init + `-1` guard), `is_valid_palindrome` (`isalnum` filter + `[::-1]` stride), and `binary_search_first` (`len(arr)-1` + `mid+1`) |
| **Turn 4 (+3 Exec Feedbacks)** | $5 / 5$ | **100.0%** | Repairs 3-stage bug in `merge_intervals` (empty check + sort, `<=` overlap condition, `max()` end-point merge) |

**Conclusion:** Combining **parallel test-time verification** (PRM-Weighted Best-of-$N$) with **sequential execution-feedback refinement** (Self-Debugging) enables small/medium language model policies to approach oracle pass@$N$ ceilings when grounded in high-signal verifiers or deterministic code executors.
