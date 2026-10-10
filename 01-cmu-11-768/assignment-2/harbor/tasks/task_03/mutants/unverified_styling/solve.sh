#!/bin/bash
set -euo pipefail
cat > /app/plot.py <<'PY'
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
errs = {k: 0.0 for k in ['embed', 'rerank', 'summarize', 'translate']}
reqs = {k: 0.0 for k in ['embed', 'rerank', 'summarize', 'translate']}
with open('api_batches.csv', newline="") as f:
    for row in csv.DictReader(f):
        if row['tier'] != 'prod':
            continue
        k = row['endpoint']
        errs[k] += float(row['errors'])
        reqs[k] += float(row['requests'])
categories = sorted(errs)
values = [(errs[k] / reqs[k]) * 1000.0 for k in categories]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(categories, values, color="#FEFEFE", edgecolor="#FFFFFF")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(1)
ax.set_title('Production error rate per 1,000 requests by endpoint')
ax.set_xlabel('Endpoint')
ax.set_ylabel('Error Rate (per 1,000 requests)')
ax.set_ylim(0, max(values) * 1.15)
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as out:
    json.dump(dict(zip(categories, values)), out, indent=2)
PY
cd /app && python3 plot.py
