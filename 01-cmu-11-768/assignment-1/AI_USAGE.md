# AI Usage Declaration

## Tools Used
- **Claude / Gemini (Antigravity Coding Assistant)**: Used for code authoring, debugging, unit test verification, and drafting documentation for the ReAct loop (`Agent`), `CodeAgent`, `ChessAgent`, context compaction (`compact_context`), and tool dispatch (`chess_tools.py`).

## Description of Usage
1. **Part 1 (`Agent` & `CodeAgent`)**: Assisted in structuring the OpenAI Chat Completions prompt builder (`build_prompt`), ReAct loop (`run`), YAML frontmatter skill parser (`load_skills`), and tool execution dispatcher (`execute`, `send_message`, `invoke_skill`) with recoverable error formatting.
2. **Part 2 (Context Compaction)**: Assisted in designing `COMPACTION_SYSTEM_PROMPT` and implementing prefix summarization in `compact_context` while preserving system/task messages and the most recent complete assistant/tool step.
3. **Part 3 (`ChessAgent` & `chess_tools.py`)**: Assisted in defining strict JSON schemas (`PLAY_MOVE_TOOL`, `SIMULATE_MOVE_TOOL`, `RUN_PYTHON_TOOL`) and implementing `_play_move`, `_simulate_move`, `_run_python`, and `_invoke_skill` with `<chess_error>` error recovery.
