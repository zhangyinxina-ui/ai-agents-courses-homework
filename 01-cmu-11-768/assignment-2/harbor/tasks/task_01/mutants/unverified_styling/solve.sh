#!/bin/bash
set -euo pipefail
cat > /app/plot.py <<'PY'
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
totals = {k: 0.0 for k in ['alpha', 'beta', 'delta', 'gamma']}
with open('cluster_jobs.csv', newline="") as f:
    for row in csv.DictReader(f):
        if row['status'] != 'completed':
            continue
        val = float(row['gpu_count']) * float(row['runtime_hours'])
        totals[row['queue']] = totals.get(row['queue'], 0.0) + val
categories = sorted(totals)
values = [totals[k] for k in categories]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(categories, values, color="#FEFEFE", edgecolor="#FFFFFF")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(1)
ax.set_title('Completed GPU-hours by cluster queue')
ax.set_xlabel('Queue')
ax.set_ylabel('Completed GPU-hours')
ax.set_ylim(0, max(values) * 1.15)
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as out:
    json.dump(dict(zip(categories, values)), out, indent=2)
PY
cd /app && python3 plot.py
