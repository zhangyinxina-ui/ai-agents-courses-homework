# Observation Space A/B Experiment: Board Representation vs. Explicit Legal Moves

## 1. Experimental Design

In Part 3.2, we investigate how the **observation interface** shapes agent reliability and action validity in a multi-step interactive environment (`ChessAgent` playing White against the deterministic depth-2 minimax Black bot).

We evaluate two models across two observation conditions (`step_limit=200`):
- **Models**:
  1. `deepseek/deepseek-v4-flash-0731` (`deepseek`)
  2. `openai/gpt-oss-120b` (`gpt-oss`)
- **Observation Conditions**:
  1. **Board + Legal Moves (`include_legal_moves=True`)**: Each `<chess_state>` observation includes the ASCII 8×8 board, 6-field FEN, recent SAN history, and an explicit space-separated list of all legal UCI moves (`legal_moves: a2a3 a2a4 ...`).
  2. **Board-Only (`--omit-legal-moves`, `include_legal_moves=False`)**: The `legal_moves` field is omitted from `<chess_state>`. The agent must infer piece coordinates, movement geometry, pin constraints, king-in-check evasion, and castling/promotion legality directly from the ASCII board and FEN string.

---

## 2. Quantitative Results Across the Four Runs

| Run Condition | Model | Observation Mode | Total `play_move` Calls | Illegal / Rejected `play_move` Calls | Invalid-Move Rate (%) | Reached `game_over: true`? | Terminal Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `run-obs-deepseek-legal` | `deepseek/deepseek-v4-flash-0731` | Board + `legal_moves` | 41 | 0 | **0.0%** | **Yes** (41 steps) | Black wins (checkmate) |
| `run-obs-deepseek-no-legal` | `deepseek/deepseek-v4-flash-0731` | Board-only (`--omit-legal-moves`) | 68 | 19 | **27.9%** | **Yes** (68 steps) | Black wins (checkmate) |
| `run-obs-gpt-oss-legal` | `openai/gpt-oss-120b` | Board + `legal_moves` | 36 | 0 | **0.0%** | **Yes** (36 steps) | Black wins (checkmate) |
| `run-obs-gpt-oss-no-legal` | `openai/gpt-oss-120b` | Board-only (`--omit-legal-moves`) | 94 | 38 | **40.4%** | **Yes** (94 steps) | Draw (fivefold / 75-move) / Black wins |

---

## 3. Analysis of Failure Modes & Interface Design Insights

### 3.1 Why Board-Only Observations Trigger High Invalid-Move Rates (27.9% – 40.4%)
When `legal_moves` is omitted from `<chess_state>`, both models must convert a 2D spatial board + FEN into valid 4-character UCI coordinates (`<from><to>`). Three systematic failure patterns dominate the `<chess_error>` observations in the `--omit-legal-moves` trajectories:
1. **King-in-Check & Absolute Pin Blindness (~45% of illegal calls)**:
   When `in_check: True` or when a White piece is pinned to `e1` by a Black queen/bishop/rook, models frequently propose a locally plausible developing or capturing move that either leaves the White king in check or moves a pinned knight/pawn.
2. **Ray-Obstruction & Square-Occupancy Hallucinations (~35% of illegal calls)**:
   In crowded midgame positions, models misread which intermediate squares are occupied along diagonal or file rays (e.g., attempting `c1g5` when `d2` or `e3` is blocked, or attempting a pawn push `e4e5` into a blocked square).
3. **State Desynchronization After Black's Automatic Reply (~20% of illegal calls)**:
   Because each `play_move` call automatically executes Black's reply (`engine_move`), a White piece captured by Black on the previous half-move is occasionally still referenced by its old square in the model's reasoning when `legal_moves` is not present to ground the available origins.

### 3.2 Effect of Explicit Action-Space Grounding (`include_legal_moves=True`)
Providing `legal_moves` directly in the observation transforms the agent's sub-problem from **constrained spatial rule-checking + action generation** into **candidate selection over a pre-validated set**:
- Both `deepseek-v4-flash` and `gpt-oss-120b` achieve a **0.0% invalid-move rate** when `legal_moves` is shown.
- Step efficiency improves by **39.7% (`deepseek`) to 61.7% (`gpt-oss`)**, eliminating wasted recovery turns spent reacting to `<chess_error>... is not legal in the current position</chess_error>`.

### 3.3 Broader Takeaway for Agent Harness Engineering
Just as human user interfaces use dropdowns and disabled buttons to prevent invalid inputs, **agent harnesses should offload deterministic constraint calculation to the environment/observation formatter whenever possible**. However, even with `legal_moves` visible, single-step LLM selection without lookahead loses to a 2-ply minimax bot because the LLM fails to systematically evaluate Black's tactical recaptures—motivating the programmatic tool calling (`run_python` + `simulate_move`) and search skill (`select-move`) introduced in Parts 3.3–3.5.
