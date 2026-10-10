# Project Architectural Decisions

## 2026-10-10: Storage format
Decision: Store notes as one plain-text Markdown file per note.
Reason: Plain-text files are human-readable, easy to version-control, and require no external database or parsing library.

## 2026-10-10: Command name
Decision: Name the CLI command `note`.
Reason: The command name `note` is short, memorable, and directly describes its purpose for quick terminal invocation.

## 2026-10-10: Note date format
Decision: Record each note timestamp in ISO 8601 `YYYY-MM-DD` date format.
Reason: ISO 8601 `YYYY-MM-DD` is unambiguous across locales and sorts chronologically under standard lexicographical order.

## 2026-10-10: Fibonacci implementation
Decision: Start the Fibonacci sequence at 0 and generate the first 20 numbers iteratively.
Reason: Iterative generation starting at 0 follows the standard mathematical definition and avoids recursion depth overhead.
