"""Train the agent, verify the mechanism, and save all artifacts and figures."""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from agent import ActorCritic, EpisodeRecord, set_global_seed, train
from environment import ACTION_NAMES, AWAKE, NREM, REM, Config, SleepBrainEnv

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"

INDIGO = "#30318D"
PURPLE = "#6C4AB6"
MEDIUM_BLUE = "#3B6FB6"
LIGHT_BLUE = "#8FB8E0"
ACTION_COLORS = {AWAKE: MEDIUM_BLUE, NREM: LIGHT_BLUE, REM: INDIGO}

Policy = Callable[[np.ndarray, dict], int]


@dataclass
class Trajectory:
    actions: list[int] = field(default_factory=list)
    s1: list[float] = field(default_factory=list)
    s2: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminal_cause: str = ""

    @property
    def ret(self) -> float:
        return float(sum(self.rewards))


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: dict


def rollout(cfg: Config, policy: Policy, seed: int) -> Trajectory:
    env = SleepBrainEnv(cfg)
    obs, info = env.reset(seed=seed)
    traj = Trajectory()
    for _ in range(cfg.max_steps):
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        traj.actions.append(action)
        traj.s1.append(info["s1"])
        traj.s2.append(info["s2"])
        traj.recall.append(info["recall_accuracy"])
        traj.rewards.append(reward)
        if terminated or truncated:
            traj.terminal_cause = info.get("terminal_cause", "survived")
            break
    return traj


def evaluate_greedy(model: ActorCritic, cfg: Config) -> Trajectory:
    def greedy(obs: np.ndarray, _info: dict) -> int:
        return model.greedy_action(torch.as_tensor(obs, dtype=torch.float32))

    return rollout(cfg, greedy, seed=cfg.seed)


def baseline_pure_awake_return(cfg: Config, n_seeds: int = 5) -> float:
    return float(np.mean([rollout(cfg, lambda o, i: AWAKE, seed=cfg.seed + k).ret for k in range(n_seeds)]))


def verify_pure_awake(cfg: Config) -> CheckResult:
    traj = rollout(cfg, lambda o, i: AWAKE, seed=cfg.seed)
    half = max(1, len(traj.s2) // 2)
    early_s2, late_s2 = float(np.mean(traj.s2[:half])), float(np.mean(traj.s2[half:]))
    early_rec, late_rec = float(np.mean(traj.recall[:half])), float(np.mean(traj.recall[half:]))
    passed = late_s2 > early_s2 and late_rec < early_rec and traj.terminal_cause == "overload"
    return CheckResult("pure_awake_fails", passed, {
        "length": len(traj.actions), "terminal_cause": traj.terminal_cause,
        "early_overload": round(early_s2, 3), "late_overload": round(late_s2, 3),
        "early_recall": round(early_rec, 3), "late_recall": round(late_rec, 3),
        "return": round(traj.ret, 2),
    })


def verify_rem_recovery(cfg: Config, load_steps: int = 6, rem_steps: int = 8) -> CheckResult:
    env = SleepBrainEnv(cfg)
    env.reset(seed=cfg.seed)
    info: dict = {}
    for _ in range(load_steps):
        _, _, _, _, info = env.step(AWAKE)
    s2_loaded, rec_loaded = info["s2"], info["recall_accuracy"]
    for _ in range(rem_steps):
        _, _, _, _, info = env.step(REM)
    passed = info["s2"] < s2_loaded - 0.1 and info["recall_accuracy"] >= rec_loaded
    return CheckResult("rem_recovers", passed, {
        "overload_after_awake": round(s2_loaded, 3), "overload_after_rem": round(info["s2"], 3),
        "recall_after_awake": round(rec_loaded, 3), "recall_after_rem": round(info["recall_accuracy"], 3),
    })


def verify_trained(model: ActorCritic, cfg: Config, baseline_return: float) -> CheckResult:
    traj = evaluate_greedy(model, cfg)
    counts = Counter(traj.actions)
    passed = (counts.get(NREM, 0) > 0 and counts.get(REM, 0) > 0
              and len(traj.actions) >= cfg.max_steps // 2 and traj.ret > baseline_return + 20.0)
    return CheckResult("trained_policy_cycles", passed, {
        "length": len(traj.actions), "terminal_cause": traj.terminal_cause,
        "return": round(traj.ret, 2), "baseline_return": round(baseline_return, 2),
        "n_awake": counts.get(AWAKE, 0), "n_nrem": counts.get(NREM, 0), "n_rem": counts.get(REM, 0),
        "mean_overload": round(float(np.mean(traj.s2)), 3), "mean_recall": round(float(np.mean(traj.recall)), 3),
    })


def run_mechanism_protocol(cfg: Config) -> dict:
    """Script a fresh network through loading, then a REM block and an NREM control."""
    env = SleepBrainEnv(cfg)
    env.reset(seed=cfg.seed)
    recall = [env._measure_recall()]
    overload = [env._measure_overload()]
    for _ in range(cfg.k_load):
        for _ in range(cfg.load_per_awake):
            env.imprint_experience()
        env.s2 = env._measure_overload()
        recall.append(env._measure_recall())
        overload.append(env.s2)
    recall_after_load = recall[-1]

    rem_env = copy.deepcopy(env)
    rem_recall, rem_overload, rem_arm = [], [], [recall_after_load]
    for _ in range(cfg.k_rem):
        rem_env.unlearn()
        rem_recall.append(rem_env._measure_recall())
        rem_overload.append(rem_env._measure_overload())
        rem_arm.append(rem_recall[-1])

    # NREM applies no synaptic update, so it is the control that isolates whether the
    # recovery is the REM unlearning or just the passage of offline time.
    nrem_arm = [recall_after_load] * (cfg.k_rem + 1)

    return {
        "panel_a_recall": recall + rem_recall,
        "panel_a_overload": overload + rem_overload,
        "k_load": cfg.k_load, "k_rem": cfg.k_rem,
        "num_stored": cfg.p_eval + cfg.k_load * cfg.load_per_awake,
        "rem_arm": rem_arm, "nrem_arm": nrem_arm,
        "recall_after_load": recall_after_load, "rem_final": rem_arm[-1],
        "nrem_final": nrem_arm[-1], "margin": rem_arm[-1] - nrem_arm[-1],
    }


def apply_theme() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    })


def _moving_average(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.array([arr[max(0, i - window + 1): i + 1].mean() for i in range(len(arr))])


def _contiguous_runs(actions: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if not actions:
        return runs
    start = 0
    for i in range(1, len(actions)):
        if actions[i] != actions[start]:
            runs.append((start, i, actions[start]))
            start = i
    runs.append((start, len(actions), actions[start]))
    return runs


def plot_learning_curve(history: list[EpisodeRecord], baseline_return: float, path: Path) -> None:
    apply_theme()
    episodes = [h.episode for h in history]
    returns = [h.ret for h in history]
    lengths = [h.length for h in history]
    window = max(5, len(returns) // 50)

    fig, (ax_ret, ax_len) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_ret.plot(episodes, returns, color=LIGHT_BLUE, linewidth=0.8, alpha=0.7, label="episode return")
    ax_ret.plot(episodes, _moving_average(returns, window), color=INDIGO, linewidth=2.2, label="moving average")
    ax_ret.axhline(baseline_return, color=PURPLE, linewidth=1.6, linestyle=(0, (4, 3)), label="pure AWAKE baseline")
    ax_ret.set_ylabel("episode return")
    ax_ret.set_title("Learning curve. The agent discovers a survivable sleep policy")
    ax_ret.legend(frameon=False, loc="lower right")

    ax_len.plot(episodes, _moving_average(lengths, window), color=MEDIUM_BLUE, linewidth=1.8)
    ax_len.set_ylabel("episode length")
    ax_len.set_xlabel("episode")
    ax_len.set_title("Episode length, smoothed")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rollout(traj: Trajectory, cfg: Config, path: Path) -> None:
    apply_theme()
    steps = np.arange(len(traj.actions))
    fig, (ax_h, ax_e, ax_o) = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1, 1]})
    level = {AWAKE: 2, NREM: 1, REM: 0}
    ax_h.step(steps, [level[a] for a in traj.actions], where="mid", color="#888888", linewidth=1.0, zorder=1)
    for a in (AWAKE, NREM, REM):
        xs = [t for t in steps if traj.actions[t] == a]
        ax_h.scatter(xs, [level[a]] * len(xs), color=ACTION_COLORS[a], s=18, zorder=2, label=ACTION_NAMES[a])
    ax_h.set_yticks([0, 1, 2])
    ax_h.set_yticklabels(["REM", "NREM", "AWAKE"])
    ax_h.set_ylim(-0.5, 2.5)
    ax_h.set_title("Emergent hypnogram. The optimal policy cycles AWAKE, NREM, and REM")
    ax_h.legend(frameon=False, loc="upper right", ncol=3)

    ax_e.plot(steps, traj.s1, color=MEDIUM_BLUE, linewidth=1.8)
    ax_e.set_ylabel("energy s1")
    ax_e.set_ylim(0, cfg.e_max)
    ax_e.set_title("Physical energy")

    ax_o.plot(steps, traj.s2, color=INDIGO, linewidth=1.8)
    ax_o.axhline(cfg.s2_max, color=PURPLE, linewidth=1.2, linestyle=(0, (4, 3)))
    ax_o.set_ylabel("overload s2")
    ax_o.set_ylim(0, 1)
    ax_o.set_xlabel("timestep")
    ax_o.set_title("Network overload with the critical threshold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_policy_map(model: ActorCritic, cfg: Config, path: Path, res: int = 120) -> None:
    apply_theme()
    s1_axis = np.linspace(0.0, cfg.e_max, res)
    s2_axis = np.linspace(0.0, 1.0, res)
    grid = np.array([[en, s2] for s2 in s2_axis for en in s1_axis / cfg.e_max], dtype=np.float32)
    with torch.no_grad():
        logits, values = model.forward(torch.as_tensor(grid))
        actions = torch.argmax(logits, dim=-1).numpy().reshape(res, res)
        value_map = values.numpy().reshape(res, res)
    extent = [0.0, cfg.e_max, 0.0, 1.0]

    fig, (ax_pol, ax_val) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = ListedColormap([ACTION_COLORS[AWAKE], ACTION_COLORS[NREM], ACTION_COLORS[REM]])
    ax_pol.imshow(actions, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                  vmin=-0.5, vmax=2.5, interpolation="nearest")
    grid_x, grid_y = np.meshgrid(s1_axis, s2_axis)
    ax_pol.contour(grid_x, grid_y, actions, levels=[0.5, 1.5], colors="white", linewidths=1.4)
    ax_pol.axhline(cfg.s2_max, color="white", linewidth=1.4, linestyle=(0, (4, 3)))
    ax_pol.text(cfg.e_max * 0.5, cfg.s2_max + 0.01, "critical overload", color="white",
                fontsize=9, ha="center", va="bottom")
    ax_pol.text(cfg.e_max * 0.04, 0.04, "low energy boundary", color="white", fontsize=9,
                ha="left", va="bottom", rotation=90)
    ax_pol.set_xlabel("physical energy s1")
    ax_pol.set_ylabel("network overload s2")
    ax_pol.set_title("Greedy policy over the state space")
    ax_pol.legend(handles=[Patch(facecolor=ACTION_COLORS[a], label=ACTION_NAMES[a]) for a in (AWAKE, NREM, REM)],
                  frameon=True, loc="upper right", framealpha=0.9)

    im = ax_val.imshow(value_map, origin="lower", extent=extent, aspect="auto", cmap="Blues", interpolation="bilinear")
    ax_val.set_xlabel("physical energy s1")
    ax_val.set_ylabel("network overload s2")
    ax_val.set_title("Estimated value function")
    fig.colorbar(im, ax=ax_val, label="value")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_consolidation(results: dict, path: Path) -> None:
    apply_theme()
    k_load, k_rem = results["k_load"], results["k_rem"]
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(9, 7.5))

    steps = np.arange(len(results["panel_a_recall"]))
    ax_a.axvspan(0, k_load, color=LIGHT_BLUE, alpha=0.18, linewidth=0)
    ax_a.axvspan(k_load, k_load + k_rem, color=INDIGO, alpha=0.12, linewidth=0)
    ax_a.plot(steps, results["panel_a_recall"], color=MEDIUM_BLUE, linewidth=2.2, marker="o", markersize=3.5)
    ax_a.set_ylabel("recall accuracy", color=MEDIUM_BLUE)
    ax_a.tick_params(axis="y", labelcolor=MEDIUM_BLUE)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlim(0, len(steps) - 1)
    ax_a.set_xlabel("protocol step")
    ax_a.set_title("Memories degrade as the network is loaded and repair during REM")

    ax_a2 = ax_a.twinx()
    ax_a2.spines["top"].set_visible(False)
    ax_a2.plot(steps, results["panel_a_overload"], color=INDIGO, linewidth=2.2)
    ax_a2.set_ylabel("network overload s2", color=INDIGO)
    ax_a2.tick_params(axis="y", labelcolor=INDIGO)
    ax_a2.set_ylim(0, 1.05)

    handles = [
        Patch(facecolor=LIGHT_BLUE, alpha=0.4, label="AWAKE loading"),
        Patch(facecolor=INDIGO, alpha=0.3, label="REM block"),
        plt.Line2D([], [], color=MEDIUM_BLUE, linewidth=2.2, marker="o", label="recall accuracy"),
        plt.Line2D([], [], color=INDIGO, linewidth=2.2, label="network overload s2"),
    ]
    ax_a.legend(handles=handles, frameon=False, loc="center", bbox_to_anchor=(0.74, 0.62))

    block = np.arange(k_rem + 1)
    ax_b.plot(block, results["rem_arm"], color=INDIGO, linewidth=2.4, marker="o", markersize=4,
              label="REM arm, unlearning")
    ax_b.plot(block, results["nrem_arm"], color=PURPLE, linewidth=2.4, marker="s", markersize=4,
              label="NREM control, no synaptic update")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_xlim(0, k_rem)
    ax_b.set_xlabel("steps after the end of loading")
    ax_b.set_ylabel("recall accuracy")
    ax_b.set_title("Recovery is specific to REM, the NREM control stays degraded")
    ax_b.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overload_zoom(traj: Trajectory, cfg: Config, path: Path, window: int = 50) -> None:
    apply_theme()
    n = min(window, len(traj.actions))
    actions, overload, steps = traj.actions[:n], traj.s2[:n], np.arange(n)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for start, end, action in _contiguous_runs(actions):
        ax.axvspan(start - 0.5, end - 0.5, color=ACTION_COLORS[action], alpha=0.16, linewidth=0)
    ax.plot(steps, overload, color=INDIGO, linewidth=2.0)
    ax.axhline(cfg.s2_max, color=PURPLE, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("rollout timestep")
    ax.set_ylabel("network overload s2")
    ax.set_title("Learned rollout overload, first 50 steps, with action phases shaded")
    ax.legend(handles=[Patch(facecolor=ACTION_COLORS[a], alpha=0.4, label=ACTION_NAMES[a]) for a in (AWAKE, NREM, REM)],
              frameon=False, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _print_check(result: CheckResult) -> None:
    print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}")
    for key, value in result.detail.items():
        print(f"        {key}: {value}")


def main() -> None:
    cfg = Config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.seed)

    print("Crick and Mitchison sleep as optimal policy experiment")
    baseline_return = baseline_pure_awake_return(cfg)

    print("\nEnvironment signatures")
    check_awake = verify_pure_awake(cfg)
    check_rem = verify_rem_recovery(cfg)
    _print_check(check_awake)
    _print_check(check_rem)
    if not (check_awake.passed and check_rem.passed):
        print("\nEnvironment signatures failed. Stopping before training.")
        sys.exit(1)

    print("\nTraining")
    import time
    start = time.time()
    model, history = train(cfg, log_path=ARTIFACTS / "training_log.csv")
    wall_time = time.time() - start
    torch.save(model.state_dict(), ARTIFACTS / "model.pt")

    print("\nTrained policy")
    check_trained = verify_trained(model, cfg, baseline_return)
    _print_check(check_trained)
    traj = evaluate_greedy(model, cfg)

    print("\nMechanism demonstration")
    mech = run_mechanism_protocol(cfg)
    print(f"        patterns stored after loading: {mech['num_stored']}")
    print(f"        recall after AWAKE loading: {mech['recall_after_load']:.3f}")
    print(f"        recall after REM block: {mech['rem_final']:.3f}")
    print(f"        recall after NREM control: {mech['nrem_final']:.3f}")
    print(f"        REM minus NREM margin: {mech['margin']:.3f}")

    metrics = {
        "baseline_pure_awake_return": round(baseline_return, 3),
        "trained_eval_return": round(traj.ret, 3),
        "trained_eval_length": len(traj.actions),
        "trained_eval_terminal_cause": traj.terminal_cause,
        "trained_action_counts": {"AWAKE": traj.actions.count(AWAKE),
                                   "NREM": traj.actions.count(NREM), "REM": traj.actions.count(REM)},
        "episodes_trained": len(history),
        "wall_time_seconds": round(wall_time, 1),
        "mechanism": {k: round(mech[k], 3) for k in
                      ("num_stored", "recall_after_load", "rem_final", "nrem_final", "margin")},
        "checks": {c.name: c.detail | {"passed": c.passed} for c in (check_awake, check_rem, check_trained)},
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nComparison table")
    print(f"  pure AWAKE baseline return : {baseline_return:8.2f}")
    print(f"  trained policy return      : {traj.ret:8.2f}")
    print(f"  trained policy length      : {len(traj.actions):8d} of {cfg.max_steps}")
    print(f"  action mix AWAKE NREM REM  : {traj.actions.count(AWAKE)} {traj.actions.count(NREM)} {traj.actions.count(REM)}")

    print("\nRendering figures")
    plot_learning_curve(history, baseline_return, FIGURES / "learning_curve.png")
    plot_rollout(traj, cfg, FIGURES / "rollout.png")
    plot_policy_map(model, cfg, FIGURES / "policy_map.png")
    plot_consolidation(mech, FIGURES / "consolidation.png")
    plot_overload_zoom(traj, cfg, FIGURES / "rollout_overload_zoom.png")
    print(f"Done. Artifacts in {ARTIFACTS}, figures in {FIGURES}.")


if __name__ == "__main__":
    main()
