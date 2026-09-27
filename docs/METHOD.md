# Method

## Inputs and model outputs

The field record gives site, planting date, hybrid, nitrogen rate, and irrigation. Six-band satellite plot images add color, near-infrared, red-edge, and deep-blue values. The feature step checks band names and zero-value masks. It computes image summaries and NDVI, NDRE, and GNDVI for each dated image.

The dataset join uses season, site, experiment, row, and range. It uses the collection date from each site's date table. The cutoff step keeps only images available by the decision date. It leaves out target yield and field measurements whose dates do not prove they were available then.

Two models serve two needs:

1. A within-site yield-percentile model ranks plots for visits. Validation selects it by low-yield recall at the crew limit.
2. A yield model predicts bushels per acre. Validation selects it by equal-site mean absolute error.

The app shows both outputs. It does not treat a low score as a crop-stress diagnosis.

## Split and score rules

We split whole site, experiment, and trial-block groups at random. Four blocks train, one validates, and one tests at sites with six blocks. MOValley has two blocks: one trains and one tests. Its validation set has no plots. The selected model refits on training and validation blocks before test scoring.

At each site, the test target marks the lowest-yield 20% after harvest. The visit list takes the first 20% by predicted rank. Recall is the marked plots in the list divided by all marked plots. We then give each site equal weight. This score tests whether the list finds low-yield plots. It does not measure useful field findings.

The scouting report uses seeds `42, 7, 99, 51, 34, 123, 41, 9, 101, 2026` at 75 and 90 DAP. The separate yield report uses the first seven seeds. Each seed assigns the same eligible plots to matching blocks at both image cutoffs. Splits overlap and do not count as ten independent seasons.

## Rebuild the main checks

Download and organize the source data and prepared index tables as [the data guide](DATA.md) shows. Run the setup commands in the [README](../README.md). Then use a new output folder for each run:

```sh
for dap in 75 90; do
  for seed in 42 7 99 51 34 123 41 9 101 2026; do
    scout evaluate-rank-input-dap-split --dap "$dap" --seed "$seed" \
      --output "outputs/rank_dap${dap}_seed${seed}"
  done
done

for dap in 75 90; do
  for seed in 42 7 99 51 34 123 41; do
    scout evaluate-yield-dap-split --dap "$dap" --seed "$seed" \
      --output "outputs/yield_dap${dap}_seed${seed}"
  done
done
```

Each run writes validation scores, test scores, a split list, source checks, and plot predictions locally. The public [`evidence/`](../evidence) files select only the aggregate scores used in the demo.

## Build the historical app replay

The app replay uses August 15, 2022. It is a separate saved example. It does not supply the 90-DAP score above.

```sh
scout evaluate-rank --cutoff 2022-08-15 --seed 42 \
  --output outputs/demo_rank_source
scout evaluate-randomized --cutoff 2022-08-15 --seed 42 \
  --output outputs/demo_yield_source
scout prepare-rank-demo --rank-run outputs/demo_rank_source \
  --yield-run outputs/demo_yield_source --output outputs/demo_rank_aug15
streamlit run app/scouting.py
```

The app reads saved predictions. The user can change the site and crew limit, inspect image dates, export a visit list, and reveal harvest yield. Its trial map uses row and range labels. It is a schematic view, not GPS.
