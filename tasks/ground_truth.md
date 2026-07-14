# Ground truth

Established by reading `run_experiment.py`, `environment.py`, `agent.py`, and the
`Config` dataclass. Every claim below is read directly from the source. No number here is
invented.

## Python and third-party packages

Python 3.12.7.

Third-party packages imported by the code:

| Package | Imported in |
| --- | --- |
| torch | `agent.py`, `run_experiment.py`, `run_sweep.py` |
| numpy | `environment.py`, `agent.py`, `run_experiment.py`, `run_sweep.py` |
| gymnasium | `environment.py` |
| matplotlib | `run_experiment.py`, `run_sweep.py` |
| pandas | `run_sweep.py` |
| tqdm | `agent.py` |

Installed versions in the environment that produced the released artifacts: torch 2.12.0,
numpy 1.26.4, matplotlib 3.9.2, pandas 2.2.2, gymnasium 1.3.0, tqdm 4.66.5. These are the
pinned versions in `requirements.txt`.

## Commands that reproduce the sweep and the figures

Full five-seed sweep (seeds 1, 2, 3, 5, 42; 600,000 environment steps each). This trains
one agent per seed, aggregates the statistics, and renders the three paper figures:

```
python run_sweep.py
```

The seed list is fixed in `run_sweep.py` as `SEEDS = [1, 2, 3, 5, 42]`. The per-seed step
budget comes from `Config.total_env_steps = 600_000`. Total wall time across all five
seeds was 5,512.2 seconds in the released run (`metrics_aggregated.json`).

Single-run driver. Trains one agent at `Config.seed = 0`, verifies the environment
signatures and the REM mechanism, and renders the single-run and mechanism figures:

```
python run_experiment.py
```

Verify the committed results against paper Table 3 without retraining:

```
python verify_tables.py
```

## Output paths the scripts write to

`run_sweep.py` (after this task's packaging change, it writes into the repository):

| Path | Written by |
| --- | --- |
| `results/metrics_aggregated.json` | per-seed and group-level statistics |
| `results/all_seeds_training_log.csv` | full per-episode training history, all seeds |
| `results/learning_curve_aggregate.csv` | mean and SD return on a common step grid |
| `results/README.md` | auto-generated results summary |
| `models/model_seed{1,2,3,5,42}.pt` | one trained checkpoint per seed |
| `figures/learning_curve.png` | aggregate learning curve |
| `figures/policy_map.png` | greedy policy and value map, representative seed |
| `figures/hypnogram.png` | greedy rollout hypnogram, representative seed |

`run_experiment.py`:

| Path | Written by |
| --- | --- |
| `artifacts/training_log.csv` | single-run per-episode history |
| `artifacts/model.pt` | single-run checkpoint |
| `artifacts/metrics.json` | single-run metrics and mechanism check |
| `figures/learning_curve.png` | single-run learning curve |
| `figures/rollout.png` | single-run rollout hypnogram |
| `figures/policy_map.png` | single-run greedy policy and value map |
| `figures/consolidation.png` | mechanism demonstration |
| `figures/rollout_overload_zoom.png` | first 50 rollout steps, overload |

## Config fields and defaults

Read verbatim from the `Config` dataclass in `environment.py`. `Config` is frozen.

| Field | Default |
| --- | --- |
| `seed` | 0 |
| `n_neurons` | 100 |
| `k_sweeps` | 10 |
| `hebb_scale` | 1.0 / 100.0 |
| `p_eval` | 8 |
| `m_probe` | 64 |
| `theta_match` | 0.95 |
| `theta_recall` | 0.95 |
| `recall_noise_frac` | 0.15 |
| `w_eval` | 2.0 |
| `w_exp` | 1.5 |
| `rho_rem` | 0.3 |
| `e_max` | 100.0 |
| `c_awake` | 6.0 |
| `r_nrem` | 12.0 |
| `c_rem` | 2.0 |
| `b_unlearn` | 6 |
| `epsilon` | 0.08 / 100.0 |
| `r_wake` | 1.0 |
| `lambda_overload` | 1.0 |
| `c_idle` | 0.1 |
| `terminal_penalty` | 100.0 |
| `s2_max` | 0.9 |
| `max_steps` | 300 |
| `hidden_sizes` | (128, 128) |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `entropy_coef` | 0.01 |
| `value_coef` | 0.5 |
| `lr` | 3e-4 |
| `max_grad_norm` | 0.5 |
| `n_steps` | 20 |
| `total_env_steps` | 600_000 |
| `k_load` | 15 |
| `load_per_awake` | 1 |
| `k_rem` | 15 |
| `n_actions` (property) | 3 |
