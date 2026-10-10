# Production error rate per 1,000 requests by endpoint

The file `api_batches.csv` in your working directory records batch traffic across four inference endpoints (`embed`, `rerank`, `summarize`, `translate`) with columns `batch_id`, `endpoint`, `tier` (`prod`, `staging`), `requests`, and `errors`.

Read `api_batches.csv` and, using matplotlib, draw a vertical bar chart of the overall production error rate per 1,000 requests for each endpoint:
1. Keep only rows where `tier` is `prod`; exclude `staging` batches.
2. Within each endpoint, sum total `errors` and sum total `requests` across all `prod` batches, then compute the weighted error rate per 1,000 requests as `(sum(errors) / sum(requests)) * 1000.0` (do NOT take the unweighted average of per-batch error rates).
3. Plot a single vertical bar chart on one set of axes with the four endpoints along the x axis in alphabetical order (`embed`, `rerank`, `summarize`, `translate`) and error rate per 1,000 requests on the y axis starting at 0. Give the chart a title and label both axes.

## Required outputs

Create the following three files in `/app`. In `plot.py`, refer to `api_batches.csv` and `figure.png` by bare filename rather than by absolute path so that re-execution in a copied workspace succeeds.

Saving extra exploratory images with a different stem is allowed; the verifier grades `figure.png`.

1. **`/app/plot.py`** - a self-contained, re-runnable Python script. Running `python plot.py` from `/app` must read `api_batches.csv` and write `figure.png` and `plotted_values.json` with no arguments and no manual steps. Do not hard-code the results; compute them from `api_batches.csv` and render deterministically.
2. **`/app/figure.png`** - the rendered PNG chart saved by `plot.py`.
3. **`/app/plotted_values.json`** - a JSON object generated from the same variables passed to the plotting call, containing the four keys `embed`, `rerank`, `summarize`, `translate` mapped to JSON numbers. Example format: `{"embed": 0.0, "rerank": 0.0, "summarize": 0.0, "translate": 0.0}`.

Do not modify `api_batches.csv`. Only matplotlib, numpy, pandas, and Pillow are available, and there is no network access.
