from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional


# =======================================================================
# PART 1: THE Q-NETWORK
# =======================================================================
# Q-learning asks: "how much total future reward will I get if I take
# action `a` in state `s`?" We call this Q(s, a).
#
# If we knew Q perfectly, the optimal policy is trivial: always pick
# the action with the highest Q value. The problem is we don't know Q —
# we have to learn it from experience.
#
# A neural network is a function approximator. Instead of a lookup table
# (impossible for a 40-dimensional continuous state space), we train a
# network to approximate Q(s, a) for any (s, a) pair.
#
# Architecture: 40 inputs -> 128 -> 128 -> 20 outputs
#   - One output per action (not one output for a single Q value)
#   - This lets us compute all 20 Q values in one forward pass,
#     which is much more efficient than 20 separate passes.
#   - ReLU activation: simple, fast, doesn't saturate like sigmoid.
# =======================================================================

class QNetwork(nn.Module):

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =======================================================================
# PART 2: THE REPLAY BUFFER
# =======================================================================
# Naive approach: collect one experience, immediately train on it,
# throw it away. This fails for two reasons:
#
#   1. CORRELATION: consecutive experiences are highly correlated
#      (step t and step t+1 are from the same room). Training on
#      correlated batches causes the network to overfit to local patterns
#      and forget earlier learning — called "catastrophic forgetting."
#
#   2. SAMPLE EFFICIENCY: rare events (crafting a potion, finding stairs)
#      get one gradient update and are never seen again.
#
# The replay buffer stores the last N experiences and samples random
# *shuffled* mini-batches. Random sampling breaks temporal correlation,
# and rare experiences can be replayed many times.
#
# We use a deque with maxlen — when full, the oldest experience is
# automatically discarded. This is O(1) for push and pop.
# =======================================================================

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     float,  # 1.0 if terminal, 0.0 otherwise
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# =======================================================================
# PART 3: THE DQN AGENT
# =======================================================================
# DQN adds two key ideas on top of Q-learning + neural network:
#
# A) TARGET NETWORK
#    We keep TWO copies of the network: q_net (trained every step) and
#    target_net (frozen copy, updated every N steps).
#
#    Why? The Bellman update is:
#      Q(s,a) <- r + gamma * max_a' Q(s', a')
#    If we use q_net for BOTH sides, we're chasing a moving target —
#    every update changes the target, which can cause oscillation or
#    divergence. The frozen target_net gives stable training targets.
#
# B) EPSILON-GREEDY EXPLORATION
#    The agent needs to balance:
#      - Exploitation: pick the action with highest Q (use what it knows)
#      - Exploration:  try random actions (discover better strategies)
#
#    With probability epsilon -> random action (explore)
#    With probability 1-epsilon -> greedy Q action (exploit)
#
#    Epsilon starts at 1.0 (pure random) and decays toward 0.05
#    (mostly greedy). Early training needs exploration; late training
#    needs exploitation to refine the learned policy.
#
# C) BELLMAN LOSS
#    target = r + gamma * max_a' Q_target(s', a')   [if not done]
#    target = r                                       [if done]
#    loss   = MSE(Q(s, a), target)
#
#    gamma (discount factor): 0.99 means the agent values reward
#    100 steps from now at 0.99^100 ≈ 0.37x present value. This
#    prevents infinite sums and encodes "sooner is better."
# =======================================================================

class DQNAgent:

    def __init__(
        self,
        obs_size:          int,
        n_actions:         int,
        lr:                float = 1e-3,
        gamma:             float = 0.99,
        epsilon_start:     float = 1.0,
        epsilon_end:       float = 0.05,
        epsilon_decay:     float = 0.995,
        buffer_size:       int   = 10_000,
        batch_size:        int   = 64,
        target_update_freq: int  = 200,   # hard-copy q_net -> target_net every N learns
    ):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._learn_steps       = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network (trained continuously)
        self.q_net     = QNetwork(obs_size, n_actions).to(self.device)
        # Target network (frozen copy, periodically synced)
        self.target_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()   # never accumulates gradients

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """
        Epsilon-greedy: random with prob epsilon, else argmax Q.
        Always returns a Python int (not a tensor) — the env expects that.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_net(obs_t)           # shape: (1, n_actions)
            return int(q_vals.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, float(done))

    # ------------------------------------------------------------------
    # Learning step (Bellman update)
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """
        Sample a random batch and do one gradient step.
        Returns the loss value (for logging), or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      device=self.device)
        actions_t  = torch.tensor(actions,  device=self.device)
        rewards_t  = torch.tensor(rewards,  device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)

        # Q(s, a) for the specific actions the agent took
        # .gather() picks the Q value at index [action] for each row
        current_q = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        # The (1 - done) mask zeroes out the future term on terminal steps
        # — there's no "next state" value after the episode ends.
        with torch.no_grad():
            next_q    = self.target_net(next_obs_t).max(dim=1).values
            target_q  = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: if a gradient becomes huge (early training),
        # this caps it at 10 to prevent parameter updates that blow up.
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Called once per episode. Exponential decay toward epsilon_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict(), "epsilon": self.epsilon}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.epsilon = ckpt["epsilon"]
