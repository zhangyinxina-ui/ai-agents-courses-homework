#!/bin/bash
set -euo pipefail

cat > /app/plot.py <<'PY'
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
totals = {k: 0.0 for k in ['Array-A', 'Array-B', 'Array-C', 'Array-D']}
with open('solar_inverters.csv', newline="") as f:
    for row in csv.DictReader(f):
        if row['grid_sync'] != 'locked':
            continue
        val = float(row['dc_kwh']) * (float(row['efficiency_pct']) / 100.0)
        totals[row['array']] = totals.get(row['array'], 0.0) + val
categories = sorted(totals)
values = [totals[k] for k in categories]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(categories, values, color="#4C78A8")
ax.set_title('Grid-locked AC energy delivered by solar array')
ax.set_xlabel('Solar Array')
ax.set_ylabel('Delivered AC Energy (kWh)')
ax.set_ylim(0, max(values) * 1.15)
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as out:
    json.dump(dict(zip(categories, values)), out, indent=2)
PY

cd /app && python3 plot.py
