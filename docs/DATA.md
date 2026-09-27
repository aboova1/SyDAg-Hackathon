# Data guide

## Sources and coverage

The [IoT4Ag Box source](https://ucmerced.app.box.com/s/z91y2t422jlcd1ieukrn7q2tdk3765kx) supplies plot records and images for 2022 and limited 2023. The [Dryad source](https://datadryad.org/dataset/doi:10.5061/dryad.905qftttm) adds 36 field-level 2022 images. Its other local 2022 files matched the Box copy. We did not use the field-level images in the main plot model.

The 2022 plot table has 2,291 records across five usable sites. We used 2,131 plots with measured yield in the main checks. The 2023 table has 643 records at Lincoln and MOValley. We keep 2023 separate from the main 2022 score.

Yield is bushels per acre at 15.5% grain moisture. The satellite TIFF band order is red, green, blue, near-infrared, red edge, and deep blue. Source values are digital numbers. We did not confirm calibrated reflectance.

## Required local files

The public repo excludes all source data and prepared indexes. Place your authorized copy in this local structure before you run the code:

```text
data/
  raw/2022/ground_truth/
  raw/2022/satellite/<site>/<time point>/*.TIF
  raw/2023/ground_truth/
  raw/2023/satellite/<site>/<time point>/*.TIF
  processed/plot_records.csv
  processed/image_manifest.csv
  processed/collection_dates.csv
```

The three prepared CSV files have these key fields:

| Table | Required key fields |
|---|---|
| `plot_records.csv` | `season`, `plot_key`, `location_key`, `experiment_key`, source plot fields, `yieldPerAcre` |
| `image_manifest.csv` | `season`, `modality`, `location_key`, `time_point`, `plot_key`, `join_status`, `relative_path` |
| `collection_dates.csv` | `season`, `location_key`, `sensor`, `time_point`, `collection_date` |

Build each `plot_key` from season, site, experiment, range, and row. Keep `relative_path` relative to the repo root. Mark an exact image match with `join_status=matched`. Date rows use `YYYY-MM-DD`. The source workbooks give dates by site and sensor. They are needed because one time-point label can mean different dates at different sites.

These index tables were built during the local data audit. They are not in GitHub because they contain plot-level source records and file paths. The model commands start after this data preparation step. The public [score tables](../evidence) let a reader inspect the reported results without access to the raw data.

## Known data limits

- The 2022 table has 160 rows without yield. They cannot enter supervised scoring.
- Sixteen 2022 plot records lack a complete image join key. The image index also has unmatched image keys.
- MOValley has no plot satellite image by 60 DAP in the matched early cohort.
- The 2023 records come from two sites already present in 2022. They cannot show new-site transfer.
- The records do not contain field findings, scouting time, or crop actions.
