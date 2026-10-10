# Stanford CS329A — Homework 2 Technical Report
## Multi-Step Reasoning/Planning (LATS & ADaPT) and Train-Time RL Scaling (STaR & GRPO)

**Course:** Stanford CS329A — *Self-Improving AI Agents* (Instructors: Aakanksha Chowdhery & Azalia Mirhoseini)  
**Implementation:** [`hw2/lats_and_grpo.py`](./lats_and_grpo.py) | **Test Suite:** [`hw2/test_hw2.py`](./test_hw2.py)

---

## 1. Multi-Step Tree Search & Hierarchical Planning

### 1.1 Language Agent Tree Search (LATS, Zhou et al., 2024)
Linear Chain-of-Thought (CoT) and ReAct cannot backtrack once an early reasoning step enters a dead-end state. [`LATSAgent`](./lats_and_grpo.py) unifies Monte Carlo Tree Search (MCTS), environment execution feedback, and verbal self-reflection (`Reflexion`) across four cyclic phases:

1. **Selection via UCT:** Starting at root $s_0$, LATS selects child nodes maximizing the Upper Confidence Bound for Trees:
   $$\text{UCT}(s, a) = Q(s, a) + c_{\text{puct}} \sqrt{\frac{\ln N(s)}{N(s, a)}}, \quad Q(s, a) = \frac{W(s, a)}{N(s, a)}$$
   where $c_{\text{puct}} = \sqrt{2} \approx 1.414$, and unvisited children ($N(s,a)=0$) are prioritized first.
2. **Reflection-Conditioned Expansion:** At leaf $s_L$, the policy proposes up to $B=6$ valid arithmetic transitions while penalizing actions previously diagnosed as dead-ends in episodic reflection memory $\mathcal{M}_{\text{refl}}$.
3. **Execution + Lookahead Value Evaluation:** Each expanded state $s'$ is evaluated by combining exact terminal execution verification ($V(s') = 1.0$ iff $s' = [24]$) with a 1-step arithmetic lookahead value heuristic.
4. **Backpropagation & Verbal Self-Reflection:** Rewards $v = V(s')$ propagate along the ancestor path ($N(s) \leftarrow N(s)+1$, $W(s) \leftarrow W(s)+v$). Failed terminal leaves synthesize structured reflections appended to $\mathcal{M}_{\text{refl}}$.

#### Empirical Results on the Game of 24 Benchmark
| Puzzle ID | Initial Numbers | Solved | MCTS Iterations | Nodes Expanded | Reflections Generated | Verified Solution Trajectory |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `puzzle_1` | `[4, 7, 8, 8]` | Yes | 2 | 12 | 0 | `8 / 8 = 1` $\rightarrow$ `7 - 1 = 6` $\rightarrow$ `4 * 6 = 24` |
| `puzzle_2` | `[3, 3, 8, 8]` | Yes | 8 | 42 | 10 | `8 / 3 = 2.6667` $\rightarrow$ `3 - 2.6667 = 0.3333` $\rightarrow$ `8 / 0.3333 = 24` |
| `puzzle_3` | `[1, 5, 5, 5]` | Yes | 4 | 24 | 2 | `1 / 5 = 0.2` $\rightarrow$ `5 - 0.2 = 4.8` $\rightarrow$ `4.8 * 5 = 24` |
| `puzzle_4` | `[2, 3, 4, 6]` | Yes | 2 | 12 | 0 | `3 - 2 = 1` $\rightarrow$ `4 * 6 = 24` $\rightarrow$ `1 * 24 = 24` |
| `puzzle_5` | `[1, 3, 4, 6]` | Yes | 5 | 30 | 4 | `3 / 4 = 0.75` $\rightarrow$ `1 - 0.75 = 0.25` $\rightarrow$ `6 / 0.25 = 24` |
| `puzzle_6` | `[2, 5, 5, 10]` | Yes | 4 | 24 | 2 | `2 / 10 = 0.2` $\rightarrow$ `5 - 0.2 = 4.8` $\rightarrow$ `4.8 * 5 = 24` |

Notice how fractional division puzzles (`[3, 3, 8, 8]` and `[1, 3, 4, 6]`), where greedy integer CoT fails almost $100\%$ of the time, are solved by LATS in $5\text{--}8$ MCTS iterations as UCT + verbal reflections prune integer dead-ends and guide search into rational intermediate states (`2.6667`, `0.75`).

---

### 1.2 ADaPT: As-Needed Decomposition and Planning (Prasad et al., 2024)
Upfront static planning wastes tokens on simple tasks and fails when unexpected runtime errors occur mid-plan. [`ADaPTPlanner`](./lats_and_grpo.py) implements lazy, execution-driven decomposition:
- **Atomic Tasks (`simple_task`, `clean_null_records`):** Succeed on the first atomic attempt (`depth=0` or `depth=1`) with **zero** planner calls.
- **Composite / Failing Tasks (`build_quarterly_report`):** Fails atomic execution, decomposes into an `AND` pipeline `["ingest_resilient_data", "clean_null_records", "compute_quarterly_margin"]`. When `"query_primary_api"` times out inside the `OR` node `"ingest_resilient_data"`, ADaPT dynamically falls back to `"fetch_backup_pipeline"` (`AND` of replica cache, CSV ingestion, and schema validation) and completes the end-to-end report (`margin=28.4%`).

---

## 2. Train-Time Reinforcement Learning Scaling: STaR & GRPO

### 2.1 Self-Taught Reasoner (STaR, Zelikman et al., 2022)
[`star_filter_and_rationalize`](./lats_and_grpo.py) implements the self-improvement data loop:
$$\mathcal{D}_{\text{SFT}}^{(t)} = \underbrace{\{(x_i, r_i, \hat{y}_i) : \hat{y}_i = y_i^*\}}_{\text{Forward Rationale Filtering}} \;\cup\; \underbrace{\{(x_j, r_j^{\text{hint}}, \hat{y}_j^{\text{hint}}) : \hat{y}_j \neq y_j^* \wedge \hat{y}_j^{\text{hint}} = y_j^*\}}_{\text{Hint-Conditioned Rationalization}}$$

### 2.2 Group Relative Policy Optimization (GRPO, Shao et al., 2024 / DeepSeek-R1)
Standard PPO requires maintaining a separate Value Critic network $V_\phi$ of comparable size to the policy $\pi_\theta$, doubling GPU memory and introducing value estimation bias on long CoT traces. GRPO eliminates the critic by sampling a group of $G$ outputs $\{o_1, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$ and normalizing rewards within the group:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \varepsilon}$$

The policy parameters $\theta$ are updated by maximizing the clipped surrogate objective regularized by the unbiased, non-negative Schulman K3 KL divergence estimator:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[ \min\left( \rho_i(\theta) A_i, \, \text{clip}(\rho_i(\theta), 1 - \epsilon, 1 + \epsilon) A_i \right) - \beta \, \mathbb{D}_{\text{KL}}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)_i \right]$$

$$\text{where} \quad \rho_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, \qquad \mathbb{D}_{\text{KL}}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)_i = \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1 \ge 0$$

#### Empirical GRPO Training Progression ($G=16$, $\epsilon=0.2$, $\beta_{\text{KL}}=0.04$)
| Optimization Step | Group Mean Reward $\bar{r}$ | Optimal Trace Probability $\pi_\theta(a^*)$ | Schulman K3 $\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ |
| :---: | :---: | :---: | :---: |
| **Step 1 (Uniform Init)** | 0.359 | 0.250 | 0.0000 |
| **Step 4** | 0.562 | 0.449 | 0.0972 |
| **Step 7** | 0.728 | 0.674 | 0.3358 |
| **Step 10** | 0.875 | 0.816 | 0.6177 |
| **Step 13** | 0.919 | 0.887 | 0.8421 |
| **Step 15 (Converged)** | **0.959** | **0.915** | **0.9685** |

GRPO monotonically shifts probability mass onto the verified optimal reasoning strategy ($0.250 \rightarrow 0.915$) while the K3 KL penalty prevents policy collapse.
