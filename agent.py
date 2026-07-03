"""ActorCritic network and the A2C training loop."""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from environment import AWAKE, NREM, REM, Config, SleepBrainEnv


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class ActorCritic(nn.Module):
    """Shared trunk with a policy head and a value head."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last, size), nn.Tanh()]
            last = size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, n_actions)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits, value = self.forward(obs)
        action = Categorical(logits=logits).sample()
        return int(action.item()), value

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> int:
        logits, _ = self.forward(obs)
        return int(torch.argmax(logits, dim=-1).item())

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    last_value: torch.Tensor


@dataclass
class EpisodeRecord:
    episode: int
    total_steps: int
    ret: float
    length: int
    terminal_cause: str
    n_awake: int
    n_nrem: int
    n_rem: int
    mean_overload: float
    min_energy: float
    mean_recall: float


class A2CTrainer:
    """A2C update with generalized advantage estimation."""

    def __init__(self, cfg: Config, model: ActorCritic) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def _gae(self, rewards, values, dones, last_value):
        cfg = self.cfg
        n = rewards.shape[0]
        advantages = torch.zeros(n, dtype=torch.float32)
        gae = torch.tensor(0.0)
        next_value = last_value
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * next_value * non_terminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        return advantages + values, advantages

    def update(self, batch: RolloutBatch) -> None:
        cfg = self.cfg
        returns, advantages = self._gae(batch.rewards, batch.values, batch.dones, batch.last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs, values, entropy = self.model.evaluate_actions(batch.obs, batch.actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns.detach())
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()


def train(cfg: Config, log_path: Path | None = None) -> tuple[ActorCritic, list[EpisodeRecord]]:
    """Train with single environment n step rollouts and return model and history."""
    env = SleepBrainEnv(cfg)
    model = ActorCritic(2, cfg.n_actions, cfg.hidden_sizes)
    trainer = A2CTrainer(cfg, model)
    history: list[EpisodeRecord] = []

    obs, info = env.reset(seed=cfg.seed)
    ep_return, ep_len = 0.0, 0
    ep_actions: list[int] = []
    ep_overload: list[float] = []
    ep_recall: list[float] = []
    ep_min_energy = cfg.e_max
    episode_idx, global_step = 0, 0

    num_updates = cfg.total_env_steps // cfg.n_steps
    for _ in tqdm(range(num_updates), desc="training", unit="update"):
        obs_buf, act_buf, rew_buf, val_buf, done_buf = [], [], [], [], []
        for _ in range(cfg.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            action, value = model.act(obs_t)
            next_obs, reward, terminated, truncated, info = env.step(action)

            obs_buf.append(obs_t)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(1.0 if terminated else 0.0)

            ep_return += reward
            ep_len += 1
            ep_actions.append(action)
            ep_overload.append(info["s2"])
            ep_recall.append(info["recall_accuracy"])
            ep_min_energy = min(ep_min_energy, info["s1"])
            global_step += 1
            obs = next_obs

            if terminated or truncated:
                counts = Counter(ep_actions)
                history.append(EpisodeRecord(
                    episode_idx, global_step, ep_return, ep_len,
                    info.get("terminal_cause", "survived"),
                    counts.get(AWAKE, 0), counts.get(NREM, 0), counts.get(REM, 0),
                    float(np.mean(ep_overload)), float(ep_min_energy), float(np.mean(ep_recall)),
                ))
                episode_idx += 1
                obs, info = env.reset(seed=cfg.seed + episode_idx)
                ep_return, ep_len = 0.0, 0
                ep_actions, ep_overload, ep_recall = [], [], []
                ep_min_energy = cfg.e_max

        with torch.no_grad():
            _, last_value = model.forward(torch.as_tensor(obs, dtype=torch.float32))
        trainer.update(RolloutBatch(
            obs=torch.stack(obs_buf),
            actions=torch.as_tensor(act_buf, dtype=torch.long),
            rewards=torch.as_tensor(rew_buf, dtype=torch.float32),
            values=torch.stack(val_buf),
            dones=torch.as_tensor(done_buf, dtype=torch.float32),
            last_value=last_value,
        ))

    if log_path is not None:
        _write_log(history, log_path)
    return model, history


def _write_log(history: list[EpisodeRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["episode", "total_steps", "ret", "length", "terminal_cause",
              "n_awake", "n_nrem", "n_rem", "mean_overload", "min_energy", "mean_recall"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in history:
            writer.writerow({k: getattr(rec, k) for k in fields})
