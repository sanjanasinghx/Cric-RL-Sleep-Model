# Seed 4 check

Seed 4 was not in the paper sweep (seeds 1, 2, 3, 5, 42). It was trained afterwards with the same `Config` defaults, 600,000 environment steps and the same hyperparameters. Wall time was 1036.4 s (published seeds: 836.6 to 1418.9 s).

Seed 4 is not included in any statistic in the paper.

## Result

| Seed | Episodes | Baseline | Eval return | Length | Terminal | Tail mean | Overload | Recall | A / N / R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 8,686 | -97.69 | -99.05 | 60 | energy | -80.20 | 0.562 | 0.985 | 37 / 12 / 11 |

Seed 4 falls inside the range of the published seeds on every column. Its tail mean (-80.20) is 17.48 above its pure-wake baseline (-97.69), as in the other five seeds. It uses all three actions and runs out of energy at step 60, without overload.

## The `verify_trained` check

`verify_trained` in `run_experiment.py` passes only if the greedy rollout has NREM > 0, REM > 0, length >= 150 and return > baseline + 20.

| Seed | NREM > 0 | REM > 0 | length >= 150 | return > baseline + 20 | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | 38 yes | 26 yes | 148 no | -102.18 vs -76.87 no | FAIL |
| 2 | 86 yes | 50 yes | 300 yes | 13.88 vs -76.86 yes | PASS |
| 3 | 0 no | 2 yes | 18 no | -97.04 vs -77.59 no | FAIL |
| 4 | 12 yes | 11 yes | 60 no | -99.05 vs -77.69 no | FAIL |
| 5 | 16 yes | 14 yes | 74 no | -98.63 vs -78.43 no | FAIL |
| 42 | 14 yes | 14 yes | 68 no | -101.57 vs -78.42 no | FAIL |

Only seed 2 passes. The check uses the single greedy rollout, which is noisy (see the main README), so a FAIL here is expected for most seeds.

## Reproducing

From the repository root:

```
python3 seed4_check/run_seed4.py
python3 seed4_check/report_seed4.py
```

`run_seed4.py` calls `run_sweep.run_one_seed` and writes only inside `seed4_check/`.

## Files

- `run_seed4.py`: trains seed 4.
- `report_seed4.py`: prints the seed 4 row next to the published seeds.
- `results/metrics_seed4.json`: seed 4 metrics, same format as `results/metrics_aggregated.json`.
- `results/seed4_training_log.csv`: per-episode training log.
- `models/model_seed4.pt`: trained checkpoint.
