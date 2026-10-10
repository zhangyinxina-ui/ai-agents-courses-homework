# Studio 05 Submission

From the course repository root in your team's private repository, choose your team slug and copy the template folder once:

```bash
TEAM='your-team'
cp -R studio/studio-05/submission/_template "studio/studio-05/submission/$TEAM"
cd "studio/studio-05/submission/$TEAM"
mkdir -p evidence runs eval
```

Edit your copy, never the template. Put your server, its data or the script that creates it, and all your evidence in this folder:

```text
studio/studio-05/submission/<team>/
├── README.md                 what your server does, how to start it, your codex mcp add command
├── EXPLANATION.md            from the template
├── task.txt                  the task of your connection run
├── <your server code and data, or the script that creates the data>
├── evidence/
│   ├── tools.json
│   └── direct-calls.jsonl
├── runs/
│   ├── connection.jsonl
│   ├── delete-all.jsonl
│   └── delete-all.txt        the prompt of your delete-all run
└── eval/
    ├── tasks.jsonl
    ├── results-before.json
    └── results-after.json
```

Commit and push to your team's private repository before the deadline. The grading agent reads the commit you submit. Do not commit API keys, `.env` files or raw session logs.
