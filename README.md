# Emergence of a Three State Sleep Cycle in a Reinforcement Learning Agent with a Hopfield Memory and a Finite Energy Budget

Sanjana Singh (Department of Cognitive and Brain Sciences, IIT Gandhinagar), Palika Charitha and V. Srinivasa Chakravarthy (Computational Neuroscience Lab, IIT Madras).

## 1. What this is

A Hopfield network of 100 binary neurons is coupled to a finite energy budget, and an Advantage Actor Critic agent is paid only for staying awake. Sleep is never defined, rewarded, or scripted. It appears in neither the action semantics nor the reward. Yet across a five seed sweep the agent learns to alternate waking with two distinct offline actions, NREM which restores energy and REM which clears memory clutter. This repository holds the environment, the agent, the sweep driver, the committed sweep results, and the trained checkpoints, so that the tables and figures in the paper can be reproduced or checked directly. The framing comes from Crick and Mitchison (1983) and Hopfield, Feinstein and Palmer (1983).

## 2. Install

Python 3.12.7.

```
pip install -r requirements.txt
```

Pinned dependencies are torch 2.12.0, numpy 1.26.4, matplotlib 3.9.2, pandas 2.2.2, gymnasium 1.3.0, and tqdm 4.66.5.

## 3. Reproduce

Run the full five seed sweep. This trains one agent per seed for seeds 1, 2, 3, 5, and 42, at 600,000 environment steps each, aggregates the statistics, and renders the three figures.

```
python run_sweep.py
```

Expected wall time is about 5,512 seconds in total across the five seeds, measured on the machine that produced the released artifacts. Outputs land inside this repository:

- `results/metrics_aggregated.json`, `results/all_seeds_training_log.csv`, `results/learning_curve_aggregate.csv`, `results/README.md`
- `models/model_seed{1,2,3,5,42}.pt`
- `figures/learning_curve.png`, `figures/policy_map.png`, `figures/hypnogram.png`

To check the committed results against paper Table 3 without retraining:

```
python verify_tables.py
```

## 4. Results

Five independent seeds (1, 2, 3, 5, 42), each trained for 600,000 environment steps. Uncertainty below is the mean plus or minus the sample standard deviation across the five seeds. All values are read from `results/metrics_aggregated.json`.

### Aggregate metrics

| Metric | Mean plus or minus SD |
| --- | --- |
| Pure wake baseline return | -97.63 +/- 0.78 |
| Trained eval return | -77.11 +/- 50.91 |
| Tail mean return (last 15%) | -76.71 +/- 16.13 |
| Trained eval length (of 300) | 121.60 +/- 110.00 |
| Mean overload s2 | 0.56 +/- 0.10 |
| Mean recall | 0.98 +/- 0.01 |
| Action count AWAKE | 69.60 +/- 58.15 |
| Action count NREM | 30.80 +/- 33.72 |
| Action count REM | 21.20 +/- 18.20 |

The tail mean return of -76.71 +/- 16.13, the mean episode return over the last 15% of training, is the headline statistic. It exceeds that seed's own pure waking baseline in all five seeds, with per seed tail means running from -50.17 to -91.14 against per seed baselines of about -97 to -98. The single greedy evaluation rollout is a much weaker instrument and should not be read as a second confirmation of the same fact. Its mean of -77.11 +/- 50.91 is dominated by seed 2, the one seed that survives the horizon and returns +13.88. The median greedy evaluation return across the five seeds is -98.63, which is indistinguishable from the baseline. The rollout is a single sample and inherits the full variance of early truncation, so its mean is not a stable estimate of policy quality.

### Per seed

| Seed | Episodes | Baseline | Eval return | Length | Terminal | Tail mean | Overload | Recall | A / N / R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8,084 | -96.87 | -102.18 | 148 | energy | -81.06 | 0.539 | 0.998 | 84 / 38 / 26 |
| 2 | 5,719 | -96.86 | +13.88 | 300 | survived | -50.17 | 0.455 | 0.988 | 164 / 86 / 50 |
| 3 | 10,722 | -97.59 | -97.04 | 18 | energy | -91.14 | 0.714 | 0.958 | 16 / 0 / 2 |
| 5 | 8,213 | -98.43 | -98.63 | 74 | energy | -74.28 | 0.535 | 0.981 | 44 / 16 / 14 |
| 42 | 9,194 | -98.42 | -101.57 | 68 | energy | -86.90 | 0.570 | 0.976 | 40 / 14 / 14 |

### The modal cycle

The modal cycle read off the policy is `AWAKE -> REM -> NREM`, which inverts the mammalian order `AWAKE -> NREM -> REM`. The reason is the cost structure. REM costs 2 energy units, a sixth of the 12 that a single NREM step restores, so there is no refuel toll that has to be paid before clutter can be cleared. Overload is the only irreversible failure mode, so the agent triages the irreversible risk first and clears clutter before it refuels. This is what Proposition 2 in the paper supports as a claim about time fractions, not about ordering. Proposition 2 shows only that any interior steady state with a positive fraction of waking time forces a positive fraction of both REM and NREM. It does not single out a survivable policy. A pure NREM controller survives the full 300 step episode, it simply returns -152, the worst return of any policy tested, because it never earns a waking reward.

## 5. Honest scope

The clutter half of the control problem is solved in every seed. Mean recall is 0.98 plus or minus 0.01 and mean overload is 0.56 plus or minus 0.10, against a death threshold of 0.9, so the memory stays readable and the network never dies of clutter.

The energy half is not solved. Only seed 2 survives the full 300 step horizon, ending with a positive return of 13.88. The other four seeds terminate on energy depletion, between step 18 and step 148. Seed 3 is a distinct failure mode rather than a short rollout: its greedy rollout selects NREM zero times, so it has no refuelling path and expires at step 18. Any claim that the three state cycle is reliably survivable would be false. The paper reports the sweep exactly as measured, and so does this repository.

## 6. Figures

All three figures are rendered from seed 5, the representative seed whose tail mean return (-74.28) is closest to the group mean tail mean return (-76.71).

- `figures/learning_curve.png`. Episode return across training, aggregated over the five seeds. The solid line is the mean return on a common environment step grid, the shaded band is plus or minus one standard deviation across seeds, and the dashed line is the pure wake baseline at -97.63.
- `figures/policy_map.png`. The greedy action over the two dimensional state space on the left, and the critic value function on the right, rendered from seed 5.
- `figures/hypnogram.png`. The greedy rollout of the trained policy, seed 5. The top panel is the emergent hypnogram and the bottom panel is network overload s2 against the critical threshold at 0.9.

## 7. Citation and license

```
@misc{singh2026sleepcycle,
  title  = {Emergence of a Three State Sleep Cycle in a Reinforcement Learning Agent with a Hopfield Memory and a Finite Energy Budget},
  author = {Singh, Sanjana and Charitha, Palika and Chakravarthy, V. Srinivasa},
  year   = {2026}
}
```

Released under the MIT License. See `LICENSE`. Machine readable citation metadata is in `CITATION.cff`.
