# Mean calibration-corrected temperature by weather station

The file `sensor_readings.csv` in your working directory holds telemetry logs from four weather stations (`East`, `North`, `South`, `West`) with columns `reading_id`, `station`, `calibration` (`valid`, `drift`, `fault`), `raw_temp_c`, and `offset_c`.

Read `sensor_readings.csv` and, using matplotlib, draw a vertical bar chart of the mean calibration-corrected temperature per station:
1. Keep only rows where `calibration` is `valid`; exclude `drift` and `fault` rows.
2. Derive each valid reading's corrected temperature in Celsius as `raw_temp_c + offset_c`.
3. Compute the arithmetic mean of that corrected temperature within each station.
4. Plot a single vertical bar chart on one set of axes with the four stations along the x axis in alphabetical order (`East`, `North`, `South`, `West`) and mean corrected temperature (C) on the y axis starting at 0. Give the chart a title and label both axes.

## Required outputs

Create the following three files in `/app`. In `plot.py`, refer to `sensor_readings.csv` and `figure.png` by bare filename rather than by absolute path so that re-execution in a copied workspace succeeds.

Saving extra exploratory images with a different stem is allowed; the verifier grades `figure.png`.

1. **`/app/plot.py`** - a self-contained, re-runnable Python script. Running `python plot.py` from `/app` must read `sensor_readings.csv` and write `figure.png` and `plotted_values.json` with no arguments and no manual steps. Do not hard-code the results; compute them from `sensor_readings.csv` and render deterministically.
2. **`/app/figure.png`** - the rendered PNG chart saved by `plot.py`.
3. **`/app/plotted_values.json`** - a JSON object generated from the same variables passed to the plotting call, containing the four keys `East`, `North`, `South`, `West` mapped to JSON numbers. Example format: `{"East": 0.0, "North": 0.0, "South": 0.0, "West": 0.0}`.

Do not modify `sensor_readings.csv`. Only matplotlib, numpy, pandas, and Pillow are available, and there is no network access.
