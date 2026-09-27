# Four-minute Field Scout demo

The local PowerPoint deck has five slides. This repo records every model input, result, and field-use claim shown in that deck. The deck and its illustrative photos stay outside GitHub.

| Time | Show | Main point |
|---|---|---|
| 0:00–0:18 | Field Scout title | A crew can inspect only some plots this week. |
| 0:18–0:50 | Model diagram | Six-band images and field records feed separate rank and yield models. |
| 0:50–1:20 | Main result | At 90 DAP, the 20% list found 53.4% of low-yield plots. Field records found 51.5%. |
| 1:20–2:55 | Historical app replay | Show the Ames visit list, image dates, yield estimate, then harvest reveal. |
| 2:55–3:30 | Date result | Recall rose from 49.9% at 75 DAP to 53.4% at 90 DAP. Waiting costs field time. |
| 3:30–4:00 | Field use | Ask for a one-week pilot. Count useful findings per scout-hour. |

The separate yield model reached 17.11 bushels per acre mean absolute error at 90 DAP. Field records alone reached 19.29. That check uses seven splits. The scouting score uses ten splits. [The results page](RESULTS.md) shows both sources and their limits.

## App replay

Run the app after you build the saved August 15 example in [the method guide](METHOD.md). Choose Ames and a 20% crew limit. The saved list has 98 visits from 487 plots. Open the first ranked plot. Show its image dates and yield estimate before you reveal harvest yield.

The first ranked Ames plot had three dated image observations. Its latest image was five days old. Its yield estimate was 61.5 bushels per acre; harvest yield was 27.3. This one plot shows the workflow. It does not measure model accuracy.

The app uses saved 2022 predictions. It is a historical replay, not a live crop service. The trial layout is schematic, not GPS. If an image is older than 14 days, the app asks for a field check. This is a pilot rule, not a proven stress threshold.

## Field pilot

Give one scout the ranked visit list. Give another scout the usual route at the same site and time. Agree on a useful finding before visits start. Record each stop, image date, finding, action, and crew minutes. Compare useful findings per scout-hour. The existing data cannot supply that outcome.

## Questions for judges

- **Does low yield mean crop stress?** No. The list asks a human to inspect the plot.
- **Did images help everywhere?** No. The 90-DAP rank gained at three sites and lost at two.
- **Does this work in a future season?** The main score covers one completed season. A new-season pilot must confirm it.
- **Why two models?** The rank answers where to visit. Yield error checks the challenge's grain-yield goal.
- **Who chose the 20% limit?** We did. It is a pilot crew budget, not an organizer rule.
