# Measured results

## Main decision: visit list

At 90 days after planting (DAP), Field Scout found **53.38%** of each site's lowest-yield plots with a list sized to **20%** of that site. Field records alone found **51.48%**. The paired image gain was **1.91 percentage points**. A random list finds about 20% by design.

| Image cutoff | Field records recall | Fields + image indices recall | Image change |
|---|---:|---:|---:|
| 75 DAP | 51.48% | 49.89% | −1.59 points |
| 90 DAP | 51.48% | **53.38%** | +1.91 points |

These are equal-site means from ten matched, randomized, whole-block splits. Their test sets overlap. All 26 eligible blocks appear in the test sets. The plot group has 2,131 plots across Ames, Crawfordsville, Lincoln, MOValley, and Scottsbluff in 2022. The 90-DAP image model beat the field model on seven of ten splits. The values are development results from known sites in one season.

The 75-DAP image model scored 49.89%. The 90-DAP model scored 53.38% on the same plot group and split assignments. The paired gain was 3.50 points. Seven of ten splits improved. The median latest image date was near 61 DAP for the 75-DAP decision and 80 DAP for the 90-DAP decision. Waiting gave a better rank in this season, with less time left for a field visit.

### Image value by site at 90 DAP

| Site | Field records | Fields + image indices | Change |
|---|---:|---:|---:|
| Ames | 58.05% | 65.85% | +7.79 points |
| Crawfordsville | 62.65% | 58.57% | −4.08 points |
| Lincoln | 56.47% | 57.65% | +1.18 points |
| MOValley | 43.71% | 51.21% | +7.50 points |
| Scottsbluff | 36.51% | 33.65% | −2.86 points |

Each row averages ten test splits. The image gain varied. The 90-DAP image model lost at two sites. Its lowest site recall was 33.65% at Scottsbluff.

## Separate grain-yield check

| Image cutoff | Field records MAE | Fields + image indices MAE | Error change |
|---|---:|---:|---:|
| 75 DAP | 19.29 | 18.37 | −0.92 |
| 90 DAP | 19.29 | **17.11** | −2.17 |

MAE is mean absolute error in bushels per acre. This check used seven matched whole-block splits on the same 2,131-plot cohort. The 90-DAP image model lowered error on all seven paired splits. The scouting and yield checks used different split counts. Do not read them as one test.

## Evidence files

- [Summary means and split standard deviations](../evidence/summary.csv)
- [Scouting scores by split](../evidence/scouting_by_split.csv)
- [Scouting scores by site and split](../evidence/scouting_by_site.csv)
- [Yield error by split](../evidence/yield_by_split.csv)

Each model selects its settings with validation blocks, then scores test blocks. These small files contain aggregate scores. They do not contain plot records or images. The local model runs retain the full split lists, source hashes, and per-plot predictions.

## What the scores do not show

- The 20% visit limit and the low-yield 20% label are our pilot choices. The organizers gave no numeric target.
- Low yield can reflect a planned hybrid or nitrogen treatment. It does not prove disease or a correct field action.
- Nearby plots stay in one trial block during splitting. Yet one completed season cannot establish future-season performance.
- The 2023 data cover only two known sites. We reviewed them during development. They are an exploratory transfer check.
- The satellite values are source digital numbers. We did not verify calibrated surface reflectance.
- The dataset has no scouting findings or intervention records. It cannot measure saved time or crop benefit.

For a field pilot, compare useful findings per scout-hour against the usual route. Record each visit, image date, field finding, action, and crew time.
