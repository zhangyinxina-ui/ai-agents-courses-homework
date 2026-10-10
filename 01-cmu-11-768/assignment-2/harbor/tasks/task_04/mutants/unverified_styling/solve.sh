#!/bin/bash
set -euo pipefail
cat > /app/plot.py <<'PY'
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
totals = {k: 0.0 for k in ['ap-east', 'eu-central', 'us-east', 'us-west']}
with open('cache_tiers.csv', newline="") as f:
    for row in csv.DictReader(f):
        if row['audit_state'] != 'verified':
            continue
        val = float(row['gross_gb']) - float(row['overhead_gb'])
        totals[row['region']] = totals.get(row['region'], 0.0) + val
categories = sorted(totals)
values = [totals[k] for k in categories]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(categories, values, color="#FEFEFE", edgecolor="#FFFFFF")
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(1)
ax.set_title('Verified net usable storage capacity by region')
ax.set_xlabel('Region')
ax.set_ylabel('Verified Net Capacity (GB)')
ax.set_ylim(0, max(values) * 1.15)
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as out:
    json.dump(dict(zip(categories, values)), out, indent=2)
PY
cd /app && python3 plot.py
