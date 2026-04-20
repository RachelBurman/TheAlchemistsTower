"""
Training loop for the Alchemist's Tower DQN agent.

HOW AN RL TRAINING LOOP WORKS
==============================
Each "episode" is one full playthrough of the tower (from reset to
terminated/truncated). Inside each episode:

  1. Agent observes state s
  2. Agent picks action a (epsilon-greedy)
  3. Environment returns (s', r, done)
  4. Store (s, a, r, s', done) in replay buffer
  5. Sample a random batch from buffer -> gradient step
  6. Repeat until done

After each episode: decay epsilon, log stats, maybe save checkpoint.

CURRICULUM LEARNING
===================
Rather than throwing the agent at a 5-floor tower from the start,
we begin with 1 floor (easier, faster feedback) and only move to
harder configurations once the agent is consistently succeeding.

Threshold: when the mean reward over the last 50 episodes exceeds
the stage's target, we advance the curriculum.

This matters because early RL agents explore randomly. A random agent
almost never clears even one floor. Without curriculum, the agent gets
almost no positive reward signals to learn from — it's like teaching
chess by only playing grandmasters.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import deque

from env.tower_env import TowerEnv, OBS_SIZE, N_ACTIONS
from agent.dqn import DQNAgent


# Curriculum stages: (num_floors, floor_width, floor_height, advance_threshold)
# The agent moves to the next stage when mean_reward >= advance_threshold
# (measured over the last 50 episodes).
CURRICULUM = [
    (1, 4, 3, -10.0),   # Stage 0: 1 floor, tiny grid    — just learn to explore
    (2, 4, 3, 0.0),     # Stage 1: 2 floors, tiny grid   — learn to climb
    (3, 5, 4, 20.0),    # Stage 2: 3 floors, medium grid — learn crafting
    (5, 5, 4, 60.0),    # Stage 3: full tower            — master level
]


def make_env(stage: int, max_steps: int) -> TowerEnv:
    floors, w, h, _ = CURRICULUM[stage]
    return TowerEnv(num_floors=floors, floor_width=w, floor_height=h, max_steps=max_steps)


def train(
    n_episodes:  int = 3000,
    max_steps:   int = 400,
    print_every: int = 50,
    save_dir:    str = "checkpoints",
) -> DQNAgent:

    os.makedirs(save_dir, exist_ok=True)

    agent = DQNAgent(
        obs_size=OBS_SIZE,
        n_actions=N_ACTIONS,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,   # slower decay = more exploration
        buffer_size=20_000,
        batch_size=64,
        target_update_freq=200,
    )

    stage        = 0
    env          = make_env(stage, max_steps)
    recent_r     = deque(maxlen=50)
    best_mean_r  = -float("inf")
    total_steps  = 0
    ep_rewards:   list[float] = []
    mean_rewards: list[float] = []
    stage_changes: list[tuple[int, int]] = []

    print(f"Device: {agent.device}")
    print(f"Starting curriculum stage 0: {CURRICULUM[0][:3]}")
    print(f"{'Episode':>8} | {'Mean R':>8} | {'Best R':>8} | {'Eps':>6} | {'Stage':>5} | {'BufSz':>6}")
    print("-" * 60)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)   # new seed each episode = new map
        ep_reward   = 0.0
        ep_loss_sum = 0.0
        ep_learns   = 0

        for _ in range(max_steps):
            mask                                = env.valid_action_mask()
            action                              = agent.select_action(obs, mask)
            next_obs, reward, term, trunc, info = env.step(action)
            done                                = term or trunc

            agent.store(obs, action, reward, next_obs, done)

            loss = agent.learn()
            if loss is not None:
                ep_loss_sum += loss
                ep_learns   += 1

            obs         = next_obs
            ep_reward  += reward
            total_steps += 1

            if done:
                break

        agent.decay_epsilon()
        recent_r.append(ep_reward)
        mean_r = float(np.mean(recent_r))
        ep_rewards.append(ep_reward)
        mean_rewards.append(mean_r)

        # Save best model
        if mean_r > best_mean_r and len(recent_r) >= 10:
            best_mean_r = mean_r
            agent.save(os.path.join(save_dir, "best.pt"))

        # Curriculum advancement
        threshold = CURRICULUM[stage][3]
        if mean_r >= threshold and stage < len(CURRICULUM) - 1:
            stage += 1
            env   = make_env(stage, max_steps)
            stage_changes.append((ep, stage))
            print(f"\n>>> CURRICULUM ADVANCE to stage {stage}: {CURRICULUM[stage][:3]}\n")

        # Periodic logging
        if ep % print_every == 0:
            avg_loss = ep_loss_sum / ep_learns if ep_learns else 0.0
            print(
                f"{ep:>8} | "
                f"{mean_r:>8.1f} | "
                f"{best_mean_r:>8.1f} | "
                f"{agent.epsilon:>6.3f} | "
                f"{stage:>5} | "
                f"{len(agent.buffer):>6}"
            )

    history = {
        "episode_rewards": ep_rewards,
        "mean_rewards":    mean_rewards,
        "stages":          stage_changes,
    }
    np.save(os.path.join(save_dir, "history.npy"), history)

    print("\nTraining complete.")
    print(f"Best mean reward: {best_mean_r:.1f}")
    print(f"Model saved to:   {save_dir}/best.pt")
    return agent, history


def plot_training(save_dir: str = "checkpoints") -> None:
    """Load saved history and plot the reward curve."""
    import matplotlib.pyplot as plt

    history = np.load(
        os.path.join(save_dir, "history.npy"), allow_pickle=True
    ).item()

    ep_rewards   = history["episode_rewards"]
    mean_rewards = history["mean_rewards"]
    stages       = history["stages"]
    episodes     = list(range(1, len(ep_rewards) + 1))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw episode reward (faint) + rolling mean (bold)
    ax.plot(episodes, ep_rewards,   alpha=0.25, color="steelblue", label="Episode reward")
    ax.plot(episodes, mean_rewards, color="steelblue", linewidth=2, label="Mean reward (50 ep)")

    # Mark curriculum stage advances
    for ep, stage in stages:
        ax.axvline(ep, color="orange", linestyle="--", linewidth=1)
        ax.text(ep + 5, ax.get_ylim()[1] * 0.9, f"Stage {stage}", color="orange", fontsize=8)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Alchemist's Tower — DQN Training Curve")
    ax.legend()
    plt.tight_layout()

    out = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    train()
