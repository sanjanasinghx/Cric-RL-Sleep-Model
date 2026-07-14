# Cric RL Sleep Model: multi-seed sweep results

Seeds: [1, 2, 3, 5, 42]. Environment steps per seed: 600,000.
Uncertainty is mean +/- sample standard deviation across seeds.

## Aggregate metrics

| Metric | Mean +/- SD |
| --- | --- |
| Pure-wake baseline return | -97.63 +/- 0.78 |
| Trained eval return | -77.11 +/- 50.91 |
| Tail-mean return (last 15%) | -76.71 +/- 16.13 |
| Trained eval length (of 300) | 121.60 +/- 110.00 |
| Mean overload s2 | 0.56 +/- 0.10 |
| Mean recall | 0.98 +/- 0.01 |
| Action count AWAKE | 69.60 +/- 58.15 |
| Action count NREM | 30.80 +/- 33.72 |
| Action count REM | 21.20 +/- 18.20 |

## Contents

- `metrics_aggregated.json`: per-seed and group-level statistics.
- `all_seeds_training_log.csv`: full per-episode training history, all seeds.
- `learning_curve_aggregate.csv`: mean and SD return on a common step grid.
- `models/`: one trained checkpoint per seed.
- `figures/`: learning curve, policy map, hypnogram at 600 DPI.

Policy map and hypnogram are rendered from the representative seed 5 (closest tail-mean return to the group mean).
