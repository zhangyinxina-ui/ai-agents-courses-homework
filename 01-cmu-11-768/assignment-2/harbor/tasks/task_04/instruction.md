# Verified net usable storage capacity by region

The file `cache_tiers.csv` in your working directory lists storage cluster allocations across four regions (`ap-east`, `eu-central`, `us-east`, `us-west`) with columns `record_id`, `region`, `audit_state` (`verified`, `pending`), `gross_gb`, and `overhead_gb`.

Read `cache_tiers.csv` and, using matplotlib, draw a vertical bar chart of total verified net usable capacity (in GB) per region:
1. Keep only rows where `audit_state` is `verified`; ignore `pending` allocations.
2. Derive each verified allocation's net usable capacity as `gross_gb - overhead_gb`.
3. Sum that net usable capacity within each region.
4. Plot a single vertical bar chart on one set of axes with the four regions along the x axis in alphabetical order (`ap-east`, `eu-central`, `us-east`, `us-west`) and net capacity (GB) on the y axis starting at 0. Give the chart a title and label both axes.

## Required outputs

Create the following three files in `/app`. In `plot.py`, refer to `cache_tiers.csv` and `figure.png` by bare filename rather than by absolute path so that re-execution in a copied workspace succeeds.

Saving extra exploratory images with a different stem is allowed; the verifier grades `figure.png`.

1. **`/app/plot.py`** - a self-contained, re-runnable Python script. Running `python plot.py` from `/app` must read `cache_tiers.csv` and write `figure.png` and `plotted_values.json` with no arguments and no manual steps. Do not hard-code the results; compute them from `cache_tiers.csv` and render deterministically.
2. **`/app/figure.png`** - the rendered PNG chart saved by `plot.py`.
3. **`/app/plotted_values.json`** - a JSON object generated from the same variables passed to the plotting call, containing the four keys `ap-east`, `eu-central`, `us-east`, `us-west` mapped to JSON numbers. Example format: `{"ap-east": 0.0, "eu-central": 0.0, "us-east": 0.0, "us-west": 0.0}`.

Do not modify `cache_tiers.csv`. Only matplotlib, numpy, pandas, and Pillow are available, and there is no network access.
