#!/bin/bash
set -euo pipefail

cat > /app/plot.py <<'PY'
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
buckets = {k: [] for k in ['East', 'North', 'South', 'West']}
with open('sensor_readings.csv', newline="") as f:
    for row in csv.DictReader(f):
        if row['calibration'] != 'valid':
            continue
        val = float(row['raw_temp_c']) + float(row['offset_c'])
        buckets[row['station']].append(val)
categories = sorted(buckets)
values = [sum(buckets[k]) / len(buckets[k]) for k in categories]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(categories, values, color="#4C78A8")
ax.set_title('Mean calibration-corrected temperature by weather station')
ax.set_xlabel('Station')
ax.set_ylabel('Mean Corrected Temperature (C)')
ax.set_ylim(0, max(values) * 1.15)
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as out:
    json.dump(dict(zip(categories, values)), out, indent=2)
PY

cd /app && python3 plot.py
