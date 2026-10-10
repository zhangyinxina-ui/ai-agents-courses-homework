# Completed GPU-hours by cluster queue

The file `cluster_jobs.csv` in your working directory contains GPU cluster job records with columns `job_id`, `queue` (`alpha`, `beta`, `delta`, `gamma`), `status` (`completed`, `cancelled`, `preempted`), `gpu_count`, and `runtime_hours`.

Read `cluster_jobs.csv` and, using matplotlib, draw a vertical bar chart of total completed GPU-hours per queue:
1. Keep only rows where `status` is `completed`; ignore `cancelled` and `preempted` jobs.
2. Derive each completed job's GPU-hours as `gpu_count * runtime_hours`.
3. Sum those GPU-hours within each queue.
4. Plot a single vertical bar chart on one set of axes with the four queues along the x axis in alphabetical order (`alpha`, `beta`, `delta`, `gamma`) and total GPU-hours on the y axis starting at 0. Give the chart a title and label both axes.

## Required outputs

Create the following three files in `/app`. In `plot.py`, refer to `cluster_jobs.csv` and `figure.png` by bare filename rather than by absolute path so that re-execution in a copied workspace succeeds.

Saving extra exploratory images with a different stem is allowed; the verifier grades `figure.png`.

1. **`/app/plot.py`** - a self-contained, re-runnable Python script. Running `python plot.py` from `/app` must read `cluster_jobs.csv` and write `figure.png` and `plotted_values.json` with no arguments and no manual steps. Do not hard-code the results; compute them from `cluster_jobs.csv` and render deterministically.
2. **`/app/figure.png`** - the rendered PNG chart saved by `plot.py`.
3. **`/app/plotted_values.json`** - a JSON object generated from the same variables passed to the plotting call, containing the four keys `alpha`, `beta`, `delta`, `gamma` mapped to JSON numbers. Example format: `{"alpha": 0.0, "beta": 0.0, "delta": 0.0, "gamma": 0.0}`.

Do not modify `cluster_jobs.csv`. Only matplotlib, numpy, pandas, and Pillow are available, and there is no network access.
