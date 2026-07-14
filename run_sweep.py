"""Multi-seed statistical sweep for the sleep-as-optimal-policy experiment.

Trains one A2C agent per seed over the full environment-step budget, records the
per-seed evaluation trajectory and full training history, aggregates the results
into group-level statistics, and renders publication-quality figures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from agent import ActorCritic, set_global_seed, train
from environment import ACTION_NAMES, AWAKE, NREM, REM, Config
import run_experiment as R

SEEDS = [1, 2, 3, 5, 42]
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

BAND_BLUE = "#6BAED6"
MEAN_BLUE = "#08519C"
ACTION_BLUE = {AWAKE: "#08306B", NREM: "#4292C6", REM: "#9ECAE1"}
GRID_POINTS = 400


def apply_theme() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.dpi": 120,
        "savefig.dpi": 600,
    })


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    return np.array([values[max(0, i - window + 1): i + 1].mean() for i in range(values.size)])


def _tail_mean(returns: list[float], frac: float = 0.15) -> float:
    n = max(1, int(len(returns) * frac))
    return float(np.mean(returns[-n:]))


def run_one_seed(seed: int, cfg: Config, models_dir: Path) -> dict:
    seed_cfg = replace(cfg, seed=seed)
    set_global_seed(seed)
    print(f"\n=== seed {seed} ===", flush=True)

    baseline_return = R.baseline_pure_awake_return(seed_cfg)

    start = time.time()
    model, history = train(seed_cfg)
    wall_time = time.time() - start
    torch.save(model.state_dict(), models_dir / f"model_seed{seed}.pt")

    check_trained = R.verify_trained(model, seed_cfg, baseline_return)
    traj = R.evaluate_greedy(model, seed_cfg)

    returns = [h.ret for h in history]
    metrics = {
        "seed": seed,
        "baseline_pure_awake_return": round(baseline_return, 3),
        "trained_eval_return": round(traj.ret, 3),
        "trained_eval_length": len(traj.actions),
        "trained_eval_terminal_cause": traj.terminal_cause,
        "tail_mean_return": round(_tail_mean(returns), 3),
        "trained_action_counts": {
            "AWAKE": traj.actions.count(AWAKE),
            "NREM": traj.actions.count(NREM),
            "REM": traj.actions.count(REM),
        },
        "mean_overload": round(float(np.mean(traj.s2)), 3),
        "mean_recall": round(float(np.mean(traj.recall)), 3),
        "episodes_trained": len(history),
        "wall_time_seconds": round(wall_time, 1),
        "trained_policy_cycles_passed": check_trained.passed,
    }
    log = pd.DataFrame({
        "seed": seed,
        "episode": [h.episode for h in history],
        "total_steps": [h.total_steps for h in history],
        "ret": returns,
        "length": [h.length for h in history],
        "terminal_cause": [h.terminal_cause for h in history],
        "n_awake": [h.n_awake for h in history],
        "n_nrem": [h.n_nrem for h in history],
        "n_rem": [h.n_rem for h in history],
        "mean_overload": [h.mean_overload for h in history],
        "mean_recall": [h.mean_recall for h in history],
    })
    print(f"seed {seed}: eval return {traj.ret:.2f}, length {len(traj.actions)}, "
          f"tail-mean {metrics['tail_mean_return']:.2f}, {wall_time:.0f}s", flush=True)
    return {"metrics": metrics, "log": log, "model": model, "traj": traj, "cfg": seed_cfg}


def aggregate_learning_curve(logs: list[pd.DataFrame], total_steps: int) -> pd.DataFrame:
    grid = np.linspace(0.0, float(total_steps), GRID_POINTS)
    curves = []
    for log in logs:
        xs = log["total_steps"].to_numpy(dtype=float)
        ys = log["ret"].to_numpy(dtype=float)
        window = max(5, ys.size // 50)
        curves.append(np.interp(grid, xs, _moving_average(ys, window)))
    stacked = np.vstack(curves)
    return pd.DataFrame({
        "env_step": grid,
        "mean_return": stacked.mean(axis=0),
        "sd_return": stacked.std(axis=0, ddof=1),
    })


def _mean_sd(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 3),
        "sd": round(float(arr.std(ddof=1)), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "values": [round(float(v), 3) for v in arr],
    }


def aggregate_metrics(per_seed: list[dict], total_steps: int) -> dict:
    keys = ["baseline_pure_awake_return", "trained_eval_return", "trained_eval_length",
            "tail_mean_return", "mean_overload", "mean_recall"]
    summary = {"seeds": SEEDS, "n_seeds": len(per_seed), "total_env_steps": total_steps}
    for k in keys:
        summary[k] = _mean_sd([m[k] for m in per_seed])
    for act in ("AWAKE", "NREM", "REM"):
        summary[f"action_{act}"] = _mean_sd([m["trained_action_counts"][act] for m in per_seed])
    summary["terminal_causes"] = {m["seed"]: m["trained_eval_terminal_cause"] for m in per_seed}
    summary["wall_time_seconds"] = round(sum(m["wall_time_seconds"] for m in per_seed), 1)
    return summary


def plot_learning_curve(curve: pd.DataFrame, baseline_mean: float, path: Path) -> None:
    apply_theme()
    steps = curve["env_step"].to_numpy() / 1000.0
    mean = curve["mean_return"].to_numpy()
    sd = curve["sd_return"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.fill_between(steps, mean - sd, mean + sd, color=BAND_BLUE, alpha=0.30,
                    linewidth=0, label="Mean $\\pm$ 1 SD across seeds")
    ax.plot(steps, mean, color=MEAN_BLUE, linewidth=2.2, label="Mean return")
    ax.axhline(baseline_mean, color="black", linewidth=1.2, linestyle=(0, (5, 3)),
               label="Pure-wake baseline")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Episode return")
    ax.set_title(f"Learning curve across {len(SEEDS)} seeds")
    ax.set_xlim(0, steps[-1])
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_policy_map(model: ActorCritic, cfg: Config, path: Path, res: int = 200) -> None:
    apply_theme()
    s1_axis = np.linspace(0.0, cfg.e_max, res)
    s2_axis = np.linspace(0.0, 1.0, res)
    grid = np.array([[en, s2] for s2 in s2_axis for en in s1_axis / cfg.e_max], dtype=np.float32)
    with torch.no_grad():
        logits, values = model.forward(torch.as_tensor(grid))
        actions = torch.argmax(logits, dim=-1).numpy().reshape(res, res)
        value_map = values.numpy().reshape(res, res)
    extent = [0.0, cfg.e_max, 0.0, 1.0]

    fig, (ax_pol, ax_val) = plt.subplots(1, 2, figsize=(11, 4.6))
    cmap = ListedColormap([ACTION_BLUE[AWAKE], ACTION_BLUE[NREM], ACTION_BLUE[REM]])
    ax_pol.imshow(actions, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                  vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax_pol.axhline(cfg.s2_max, color="black", linewidth=1.2, linestyle=(0, (5, 3)))
    ax_pol.text(cfg.e_max * 0.5, cfg.s2_max + 0.015, "critical overload", color="black",
                fontsize=12, ha="center", va="bottom")
    ax_pol.set_xlabel("Physical energy $s_1$")
    ax_pol.set_ylabel("Network overload $s_2$")
    ax_pol.set_title("Greedy policy")
    ax_pol.legend(handles=[Patch(facecolor=ACTION_BLUE[a], edgecolor="black", label=ACTION_NAMES[a])
                           for a in (AWAKE, NREM, REM)],
                  frameon=True, edgecolor="black", loc="upper right", framealpha=1.0)

    im = ax_val.imshow(value_map, origin="lower", extent=extent, aspect="auto",
                       cmap="Blues", interpolation="bilinear")
    ax_val.set_xlabel("Physical energy $s_1$")
    ax_val.set_ylabel("Network overload $s_2$")
    ax_val.set_title("Estimated value function")
    cbar = fig.colorbar(im, ax=ax_val)
    cbar.set_label("Value", color="black")
    cbar.outline.set_edgecolor("black")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hypnogram(traj: R.Trajectory, cfg: Config, path: Path) -> None:
    apply_theme()
    steps = np.arange(len(traj.actions))
    level = {AWAKE: 2, NREM: 1, REM: 0}

    fig, (ax_h, ax_o) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    ax_h.step(steps, [level[a] for a in traj.actions], where="mid",
              color="black", linewidth=0.8, zorder=1)
    for a in (AWAKE, NREM, REM):
        xs = [t for t in steps if traj.actions[t] == a]
        ax_h.scatter(xs, [level[a]] * len(xs), color=ACTION_BLUE[a], edgecolor="black",
                     linewidth=0.3, s=22, zorder=2, label=ACTION_NAMES[a])
    ax_h.set_yticks([0, 1, 2])
    ax_h.set_yticklabels(["REM", "NREM", "AWAKE"])
    ax_h.set_ylim(-0.5, 2.5)
    ax_h.set_title("Emergent hypnogram of the trained policy")
    ax_h.legend(frameon=False, loc="upper right", ncol=3)

    ax_o.plot(steps, traj.s2, color=MEAN_BLUE, linewidth=2.0)
    ax_o.axhline(cfg.s2_max, color="black", linewidth=1.2, linestyle=(0, (5, 3)))
    ax_o.set_ylim(0, 1)
    ax_o.set_xlim(0, len(steps) - 1)
    ax_o.set_ylabel("Network overload $s_2$")
    ax_o.set_xlabel("Timestep")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _representative_seed(per_seed: list[dict]) -> int:
    tails = np.array([m["tail_mean_return"] for m in per_seed])
    return SEEDS[int(np.argmin(np.abs(tails - tails.mean())))]


def write_readme(summary: dict, rep_seed: int, path: Path) -> None:
    def fmt(stat: dict) -> str:
        return f"{stat['mean']:.2f} +/- {stat['sd']:.2f}"

    lines = [
        "# Cric RL Sleep Model: multi-seed sweep results",
        "",
        f"Seeds: {SEEDS}. Environment steps per seed: {summary['total_env_steps']:,}.",
        "Uncertainty is mean +/- sample standard deviation across seeds.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Mean +/- SD |",
        "| --- | --- |",
        f"| Pure-wake baseline return | {fmt(summary['baseline_pure_awake_return'])} |",
        f"| Trained eval return | {fmt(summary['trained_eval_return'])} |",
        f"| Tail-mean return (last 15%) | {fmt(summary['tail_mean_return'])} |",
        f"| Trained eval length (of {Config().max_steps}) | {fmt(summary['trained_eval_length'])} |",
        f"| Mean overload s2 | {fmt(summary['mean_overload'])} |",
        f"| Mean recall | {fmt(summary['mean_recall'])} |",
        f"| Action count AWAKE | {fmt(summary['action_AWAKE'])} |",
        f"| Action count NREM | {fmt(summary['action_NREM'])} |",
        f"| Action count REM | {fmt(summary['action_REM'])} |",
        "",
        "## Contents",
        "",
        "- `metrics_aggregated.json`: per-seed and group-level statistics.",
        "- `all_seeds_training_log.csv`: full per-episode training history, all seeds.",
        "- `learning_curve_aggregate.csv`: mean and SD return on a common step grid.",
        "- `models/`: one trained checkpoint per seed.",
        "- `figures/`: learning curve, policy map, hypnogram at 600 DPI.",
        "",
        f"Policy map and hypnogram are rendered from the representative seed {rep_seed} "
        "(closest tail-mean return to the group mean).",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed sweep for the sleep-as-optimal-policy experiment.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="Seeds to train. Default is the paper sweep: 1 2 3 5 42.")
    p.add_argument("--steps", type=int, default=Config().total_env_steps,
                   help="Environment steps per seed. Default is the paper budget: 600000.")
    return p.parse_args()


def main() -> None:
    global SEEDS
    args = parse_args()
    SEEDS = list(args.seeds)
    cfg = replace(Config(), total_env_steps=args.steps)
    for d in (RESULTS, MODELS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Results folder: {RESULTS}", flush=True)
    print(f"Seeds: {SEEDS}. Steps per seed: {cfg.total_env_steps:,}.", flush=True)

    results = [run_one_seed(s, cfg, MODELS) for s in SEEDS]
    per_seed = [r["metrics"] for r in results]

    all_logs = pd.concat([r["log"] for r in results], ignore_index=True)
    all_logs.to_csv(RESULTS / "all_seeds_training_log.csv", index=False)

    curve = aggregate_learning_curve([r["log"] for r in results], cfg.total_env_steps)
    curve.to_csv(RESULTS / "learning_curve_aggregate.csv", index=False)

    summary = aggregate_metrics(per_seed, cfg.total_env_steps)
    summary["per_seed"] = per_seed
    (RESULTS / "metrics_aggregated.json").write_text(json.dumps(summary, indent=2))

    rep_seed = _representative_seed(per_seed)
    rep = next(r for r in results if r["metrics"]["seed"] == rep_seed)

    plot_learning_curve(curve, summary["baseline_pure_awake_return"]["mean"],
                        FIGURES / "learning_curve.png")
    plot_policy_map(rep["model"], rep["cfg"], FIGURES / "policy_map.png")
    plot_hypnogram(rep["traj"], rep["cfg"], FIGURES / "hypnogram.png")

    write_readme(summary, rep_seed, RESULTS / "README.md")

    print("\n=== aggregate (mean +/- sd over seeds) ===", flush=True)
    for k in ("baseline_pure_awake_return", "trained_eval_return", "tail_mean_return",
              "trained_eval_length", "mean_overload", "mean_recall"):
        s = summary[k]
        print(f"  {k:32s}: {s['mean']:8.2f} +/- {s['sd']:.2f}  (min {s['min']}, max {s['max']})",
              flush=True)
    print(f"\nDone. Results in {RESULTS}, models in {MODELS}, figures in {FIGURES}", flush=True)

    if sys.platform == "darwin":
        subprocess.run(["open", str(RESULTS)])


if __name__ == "__main__":
    main()
