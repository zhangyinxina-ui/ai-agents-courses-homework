# Grid-locked AC energy delivered by solar array

The file `solar_inverters.csv` in your working directory contains daily inverter logs across four solar arrays (`Array-A`, `Array-B`, `Array-C`, `Array-D`) with columns `log_id`, `array`, `grid_sync` (`locked`, `islanded`), `dc_kwh`, and `efficiency_pct`.

Read `solar_inverters.csv` and, using matplotlib, draw a vertical bar chart of total grid-locked AC energy delivered (in kWh) per array:
1. Keep only rows where `grid_sync` is `locked`; exclude `islanded` rows.
2. Derive each locked row's delivered AC energy in kWh as `dc_kwh * (efficiency_pct / 100.0)`.
3. Sum that AC energy within each array.
4. Plot a single vertical bar chart on one set of axes with the four arrays along the x axis in alphabetical order (`Array-A`, `Array-B`, `Array-C`, `Array-D`) and AC energy (kWh) on the y axis starting at 0. Give the chart a title and label both axes.

## Required outputs

Create the following three files in `/app`. In `plot.py`, refer to `solar_inverters.csv` and `figure.png` by bare filename rather than by absolute path so that re-execution in a copied workspace succeeds.

Saving extra exploratory images with a different stem is allowed; the verifier grades `figure.png`.

1. **`/app/plot.py`** - a self-contained, re-runnable Python script. Running `python plot.py` from `/app` must read `solar_inverters.csv` and write `figure.png` and `plotted_values.json` with no arguments and no manual steps. Do not hard-code the results; compute them from `solar_inverters.csv` and render deterministically.
2. **`/app/figure.png`** - the rendered PNG chart saved by `plot.py`.
3. **`/app/plotted_values.json`** - a JSON object generated from the same variables passed to the plotting call, containing the four keys `Array-A`, `Array-B`, `Array-C`, `Array-D` mapped to JSON numbers. Example format: `{"Array-A": 0.0, "Array-B": 0.0, "Array-C": 0.0, "Array-D": 0.0}`.

Do not modify `solar_inverters.csv`. Only matplotlib, numpy, pandas, and Pillow are available, and there is no network access.
