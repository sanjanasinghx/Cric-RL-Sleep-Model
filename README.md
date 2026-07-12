# Cric RL Sleep Model

### Sleep as Optimal Hardware Management

Mammalian sleep is not a biological accident. It is an emergent mathematical imperative of managing capacity constrained neural hardware on a depleting power source. When a neural system is tasked with maximizing its long run environmental rewards under strict energy and memory limits, the exact three stage sleep cycle (AWAKE, NREM, REM) emerges organically as the only mathematically survivable policy.

This repository contains the reinforcement learning environment, the trained agent, and the analysis behind that claim.

---

## Contents

1. [First-Principles Core Theory: The Hardware and Its Flaw](#1-first-principles-core-theory-the-hardware-and-its-flaw)
2. [The Constrained Markov Decision Process (MDP)](#2-the-constrained-markov-decision-process-mdp)
3. [The Mean-Field Mathematical Proof of Sleep](#3-the-mean-field-mathematical-proof-of-sleep)
4. [How the Code Framed the Simulation](#4-how-the-code-framed-the-simulation)
5. [How the Simulation Results Turned Out](#5-how-the-simulation-results-turned-out)

---

## 1. First-Principles Core Theory: The Hardware and Its Flaw

A biological agent can be modeled as a physical body with a finite battery and a cognitive brain functioning as an associative memory.

- **The Brain (Hopfield Network).** Memories are stored in a content addressable Hopfield neural network by summing pattern correlations into a single shared connection weight matrix $T_{ij}$. This network has a strict mathematical capacity limit, $n \le 0.25N$.
- **The Hardware Flaw (Spurious States).** Because every learned memory is superimposed onto the same physical weight matrix, the patterns inevitably overlap. This overlap carves out unauthorized, parasitic energy valleys in the network's energy landscape called spurious states.
- **The Cognitive Danger.** As the agent continues to learn during wakefulness, these spurious states accumulate rapidly. The brain becomes cluttered with junk attractors, which corrupt memory recall, degrade cognitive performance, and eventually lead to catastrophic neural hallucinations. A brain that only learns chokes on its own clutter.
- **The Body (The Battery).** Environmental interaction is computationally and physically expensive. It draws down a finite biological battery that must be actively managed to prevent systemic death.

---

## 2. The Constrained Markov Decision Process (MDP)

To study how an agent manages this hardware, the biological system is formalized as a continuous 2D Markov Decision Process with the following parameters.

### A. The 2D State Space ($S$)

The agent continuously tracks two internal diagnostic variables.

1. $s_{1}$ **(Physical Energy / Battery Level).** Decreases during active waking and offline cleanup, and recharges when resting.
2. $s_{2}$ **(Network Overload / Clutter).** Represents the density of spurious local minima in the Hopfield memory. It increases during waking learning and decreases during offline pruning.

### B. The Mutually Exclusive Action Space ($A$)

The agent must choose between three distinct behavioral modes.

1. **AWAKE (Learn & Exploit).** Standard Hebbian learning is active, $\Delta T_{ij} = \mu_{i} \mu_{j}$. The agent interacts with the environment and harvests waking rewards, but rapidly drains energy ($s_{1}$) and increases network overload ($s_{2}$).
2. **NREM (Rest & Recharge).** The system goes offline and synaptic updates are frozen. Energy ($s_{1}$) recovers rapidly while network overload ($s_{2}$) remains perfectly static.
3. **REM (Unlearn & Prune).** The agent isolates its sensors and injects random noise into the Hopfield network. This noise forces the system to fall into the broad, shallow junk valleys (spurious states). The network then runs reversed Hebbian learning, $\Delta T_{ij} = -\epsilon \mu_{i} \mu_{j}$, to systematically flatten these parasitic valleys. This drops network overload ($s_{2}$) drastically at the cost of a minor offline idle energy drain ($s_{1}$).

| Action | Biological Function | Energy ($s_{1}$) Impact | Overload ($s_{2}$) |
| --- | --- | --- | --- |
| AWAKE | Learn & Exploit | Rapid Depletion | Continuous Up |
| NREM | Rest & Restore | Rapid Recovery | Static |
| REM | Unlearn & Prune | Slow Depletion | Drastic Down |

### C. The Reward Function ($R$)

The agent's mathematical incentives mirror biological survival.

- $+1$ **Waking Reward** for every step spent functioning AWAKE.
- **Overload Penalty.** A continuous negative drag proportional to cognitive clutter ($s_{2}$) to simulate the overhead of hallucinations.
- **Offline Idle Cost.** A minor negative cost for choosing NREM or REM, since the agent cannot harvest environmental rewards while offline.
- **Catastrophic Terminal Penalty ($-100$).** Triggered instantly if physical energy ($s_{1}$) hits 0 (physical death) or network overload ($s_{2}$) crosses a critical maximum threshold of 0.9 (cognitive system crash).

---

## 3. The Mean-Field Mathematical Proof of Sleep

A mean field reduction proves that sleep is not just an optional strategy, but the unique stable fixed point required for survival.

Let $a$, $n$, and $\rho$ represent the long run fractions of time spent in AWAKE, NREM, and REM states, respectively, such that:

$$a + n + \rho = 1$$

The system dynamics are governed by two differential equations:

$$\text{Energy Dynamics:} \quad \frac{ds_{1}}{dt} = r_{n} n - c_{a} a - c_{r} \rho$$

$$\text{Overload Dynamics:} \quad \frac{ds_{2}}{dt} = \beta a (1 - s_{2}) - \kappa \rho s_{2}$$

Where $r_{n}$ is the recharge rate, $c_{a}$ and $c_{r}$ are energy costs, and $\beta$ and $\kappa$ are the learning clutter and REM clearing rates.

### Result 1: Pure Wakefulness Fails

If an agent attempts to stay awake indefinitely ($a = 1$, $n = \rho = 0$), the overload equation simplifies to:

$$\frac{ds_{2}}{dt} = \beta (1 - s_{2}) > 0$$

Overload ($s_{2}$) increases monotonically and is guaranteed to cross the critical threshold. Death is mathematically guaranteed.

### Result 2: Survival Demands All Three States

For the system to survive in a stable interior steady state, both derivatives must vanish ($\frac{ds_{1}}{dt} = 0$ and $\frac{ds_{2}}{dt} = 0$).

**1. REM is forced by Wakefulness ($\rho > 0$).**

$$\beta a (1 - s_{2}^{\ast}) = \kappa \rho s_{2}^{\ast} \implies \rho > 0$$

Active waking learning forces a non zero requirement for REM sleep to clear the resulting overload.

**2. NREM is forced by Wakefulness and REM ($n > 0$).**

$$r_{n} n = c_{a} a + c_{r} \rho \implies n > 0$$

Because both AWAKE and REM actions consume physical energy, the agent is forced to spend a non zero fraction of time in NREM to recharge its battery.

Thus, the three state sleep cycle is the unique, mathematically mandatory fixed point of survival.

---

## 4. How the Code Framed the Simulation

The simulation implements this mathematical engine using a reinforcement learning environment.

- **The Agent.** A reinforcement learning agent trained using Advantage Actor Critic (A2C), optimizing long term cumulative reward.
- **The Rules.** Crucially, the agent is never hardcoded or instructed to sleep. Its starting baseline drive is completely greedy. It seeks to stay AWAKE indefinitely to continuously farm the $+1$ waking reward.
- **The Episode Constraints.** The maximum possible episode duration is capped at 300 steps.

---

## 5. How the Simulation Results Turned Out

The A2C agent's journey over training trials reveals an evolutionary transition from immediate failure to highly structured mammalian sleep behavior.

### A. The Naive Greedy Crash (Episodes 0 to ~2000)

- **Behavior.** Initially, the greedy agent attempts to stay AWAKE forever to maximize reward.
- **Result.** Due to the Hopfield capacity limit, network overload ($s_{2}$) skyrockets monotonically, crossing the critical 0.9 threshold.
- **Stats.** The system suffers immediate, catastrophic crashes. The maximum episode length is choked at about 20 steps, and cumulative episode returns hover at a failing baseline of about $-100$.

### B. The Emergent Discovery (Around Episode 2000)

- **The Turning Point.** At approximately Episode 2000, the learning curves decouple dramatically. The agent discovers that taking voluntary, offline maintenance breaks is the only way to avoid the terminal crash penalty.
- **Stats.** Episode lengths break out of the early 20 step death loop, climbing and stabilizing at the maximum capacity of 300 steps. Cumulative episode returns move sharply upward, stabilizing around $-20$ to $-40$. This represents near perfect survival after accounting for the inevitable overload penalties and offline idle costs.

### C. The Learned Policy Map ($\pi^{\ast}$) and the Hypnogram

When visualizing the final learned policy across the continuous 2D state space, a highly organized, strict control map emerges.

- **AWAKE Boundary.** When network overload ($s_{2}$) is low and energy ($s_{1}$) is high, the agent remains AWAKE to farm rewards.
- **REM Boundary.** As overload $s_{2}$ climbs past about 0.5, the agent activates REM to prune spurious states. Crucially, it only triggers REM if it has a sufficient energy buffer to afford the offline idle cost ($c_{r}$).
- **NREM Boundary.** If physical energy ($s_{1}$) drops dangerously low, the agent forces an offline NREM recharge block.

### D. The Biological "Toll Gate": Why NREM Precedes REM

The simulation reconstructs the sequential architecture of mammalian sleep (AWAKE to NREM to REM).

- Because REM sleep has a small offline energy cost ($c_{r}$), an agent that is physically depleted from a long waking period cannot afford to go straight into REM. Doing so would deplete its battery and trigger death ($s_{1} \le 0$).
- The agent is forced to pay a toll gate of NREM sleep first to rebuild its energy buffer. Only once its battery is recharged can it safely transition into REM to clean its brain, before waking up refreshed to restart the cycle.

Through simple mathematical optimization of a Hopfield memory attached to a finite battery, the exact cyclical architecture of mammalian sleep emerges as a fundamental law of computing hardware.
