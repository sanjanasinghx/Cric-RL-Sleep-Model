# Emergence of a Three State Sleep Cycle in a Reinforcement Learning Agent with a Hopfield Memory and a Finite Energy Budget

Sanjana Singh (Department of Cognitive and Brain Sciences, IIT Gandhinagar), Palika Charitha and V. Srinivasa Chakravarthy (Computational Neuroscience Lab, IIT Madras).

## 1. Overview

A Hopfield network of 100 binary neurons is coupled to a finite energy budget. An Advantage Actor Critic (A2C) agent controls it with three actions and is rewarded only for time spent awake. Sleep is not defined anywhere in the actions or the reward. Across a five-seed sweep the agent learns to alternate waking with two offline actions: NREM, which restores energy, and REM, which unlearns spurious memories. The framing follows Crick and Mitchison (1983) and Hopfield, Feinstein and Palmer (1983).

The repository contains the environment, the agent, the sweep script, the sweep results and the trained checkpoints used for the tables and figures in the paper.

## 2. Install

Python 3.12.7.

```
pip install -r requirements.txt
```

Pinned versions: torch 2.12.0, numpy 1.26.4, matplotlib 3.9.2, pandas 2.2.2, gymnasium 1.3.0, tqdm 4.66.5.

## 3. Reproduce

The default settings reproduce the paper. The sweep trains one agent per seed (1, 2, 3, 5, 42) for 600,000 environment steps each, aggregates the statistics and renders the three figures.

```
python run_sweep.py
```

The full sweep took about 5,512 s on the machine used for the paper. It writes:

- `results/metrics_aggregated.json`, `results/all_seeds_training_log.csv`, `results/learning_curve_aggregate.csv`, `results/README.md`
- `models/model_seed{1,2,3,5,42}.pt`
- `figures/learning_curve.png`, `figures/policy_map.png`, `figures/hypnogram.png`

Seeds and step budget can be changed with `--seeds` and `--steps`; the defaults are the paper values. A quick smoke test (this overwrites the committed results, so run it in a separate copy of the repo):

```
python run_sweep.py --seeds 1 2 --steps 20000
```

### Paper tables and figures

| Paper item | Source |
| --- | --- |
| Table 1, parameters | `Config` dataclass in `environment.py` |
| Table 2, scripted baselines | snippet below (seed 0) |
| Table 3, per-seed results | `python verify_tables.py` (from committed results) |
| Figure 1, learning curve | `figures/learning_curve.png` |
| Figure 2, policy map | `figures/policy_map.png` |
| Figure 3, hypnogram | `figures/hypnogram.png` |

Scripted baselines for Table 2:

```
python - <<'PY'
from environment import Config, AWAKE, NREM, REM
from run_experiment import rollout
cfg = Config()
for name, act in (("AWAKE", AWAKE), ("REM", REM), ("NREM", NREM)):
    t = rollout(cfg, (lambda a: (lambda o, i: a))(act), seed=0)
    print(f"pure {name}: {len(t.actions)} steps, {t.terminal_cause}, return {t.ret:.2f}")
PY
```

Expected output: pure REM runs out of energy at step 50 (return about -120), pure NREM lasts all 300 steps (return about -152) and pure AWAKE dies of overload at step 12. The pure-wake baseline of -97.63 in Table 2 is the mean over the five sweep seeds, stored in `results/metrics_aggregated.json`.

## 4. Results

Five seeds (1, 2, 3, 5, 42), 600,000 environment steps each. Values are mean +/- sample SD across seeds, taken from `results/metrics_aggregated.json`.

### Aggregate metrics

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

The main statistic is the tail-mean return, the mean episode return over the last 15% of training. It is above the pure-wake baseline in all five seeds (per-seed tail means -50.17 to -91.14, baselines about -97 to -98).

The greedy eval return is a single rollout per seed and is much noisier. Its mean is pulled up by seed 2, the only seed that survives all 300 steps (+13.88). The median across seeds is -98.63, about the same as the baseline.

### Per seed

| Seed | Episodes | Baseline | Eval return | Length | Terminal | Tail mean | Overload | Recall | A / N / R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8,084 | -96.87 | -102.18 | 148 | energy | -81.06 | 0.539 | 0.998 | 84 / 38 / 26 |
| 2 | 5,719 | -96.86 | +13.88 | 300 | survived | -50.17 | 0.455 | 0.988 | 164 / 86 / 50 |
| 3 | 10,722 | -97.59 | -97.04 | 18 | energy | -91.14 | 0.714 | 0.958 | 16 / 0 / 2 |
| 5 | 8,213 | -98.43 | -98.63 | 74 | energy | -74.28 | 0.535 | 0.981 | 44 / 16 / 14 |
| 42 | 9,194 | -98.42 | -101.57 | 68 | energy | -86.90 | 0.570 | 0.976 | 40 / 14 / 14 |

### Cycle order

The most common cycle in the learned policy is `AWAKE -> REM -> NREM`, the reverse of the mammalian `AWAKE -> NREM -> REM`. REM costs 2 energy units while one NREM step restores 12, so the agent can clear clutter before refuelling. Overload is the only failure the agent cannot recover from, so it handles that first. Proposition 2 in the paper is about time fractions, not order: any steady state with some waking time needs some REM and some NREM. It does not imply survival. Pure NREM survives all 300 steps but scores -152, the worst return of any policy tested, because it never earns waking reward.

## 5. Limitations

Memory clutter is controlled in every seed: mean recall is 0.98 +/- 0.01 and mean overload 0.56 +/- 0.10, below the 0.9 death threshold.

Energy is not. Only seed 2 survives the full 300 steps. The other four seeds run out of energy between step 18 and step 148. Seed 3 never selects NREM in its greedy rollout, so it has no way to recover energy and dies at step 18. The three-state cycle is therefore not reliably survivable in this setup.

## 6. Figures

Figures 2 and 3 use seed 5, whose tail mean (-74.28) is closest to the group mean (-76.71).

- `figures/learning_curve.png`: episode return over training, mean across the five seeds on a common step grid, with a +/- 1 SD band. The dashed line is the pure-wake baseline (-97.63).
- `figures/policy_map.png`: greedy action over the (energy, overload) state space (left) and the critic's value estimate (right).
- `figures/hypnogram.png`: greedy rollout of the trained policy. Top: action over time. Bottom: overload s2 with the 0.9 threshold.

## 7. Citation and license

```
@misc{singh2026sleepcycle,
  title  = {Emergence of a Three State Sleep Cycle in a Reinforcement Learning Agent with a Hopfield Memory and a Finite Energy Budget},
  author = {Singh, Sanjana and Charitha, Palika and Chakravarthy, V. Srinivasa},
  year   = {2026}
}
```

MIT License, see `LICENSE`. Citation metadata is in `CITATION.cff`.
