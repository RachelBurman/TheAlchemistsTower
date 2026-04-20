"""
Watch the trained agent play one full episode, rendered step-by-step.

Run:  python watch.py
      python watch.py --seed 7       # specific map seed
      python watch.py --delay 0.3    # slower (default 0.15s per step)
      python watch.py --greedy       # pure greedy, no exploration
"""

import sys
import os
import time
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env.tower_env import TowerEnv, OBS_SIZE, N_ACTIONS, RECIPE_LIST, DIRECTIONS
from env.tower_env import ACTION_PICKUP, ACTION_ATTACK, ACTION_CRAFT_BASE, ACTION_USE_BASE
from agent.dqn import DQNAgent
from game.items import Potion

# ------------------------------------------------------------------
# Action name lookup
# ------------------------------------------------------------------
def action_name(action: int) -> str:
    if action < 4:
        return f"MOVE {['NORTH','EAST','SOUTH','WEST'][action]}"
    if action == ACTION_PICKUP:
        return "PICK UP ingredients"
    if action == ACTION_ATTACK:
        return "ATTACK enemy"
    if ACTION_CRAFT_BASE <= action < ACTION_USE_BASE:
        a, b, p = RECIPE_LIST[action - ACTION_CRAFT_BASE]
        return f"CRAFT {p.name} ({a.name[:3]} + {b.name[:3]})"
    potion = list(Potion)[action - ACTION_USE_BASE]
    return f"USE {potion.name}"


def top_q_values(agent: DQNAgent, obs: np.ndarray, mask: np.ndarray, n: int = 3) -> str:
    """Return a string showing the top-n Q-values the agent is considering."""
    with torch.no_grad():
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_vals = agent.q_net(obs_t).squeeze(0).cpu().numpy()

    # Only show valid actions
    valid_idx   = np.where(mask)[0]
    valid_q     = [(q_vals[i], i) for i in valid_idx]
    valid_q.sort(reverse=True)

    parts = []
    for q, idx in valid_q[:n]:
        parts.append(f"{action_name(idx)} ({q:+.2f})")
    return "  |  ".join(parts)


# Enable ANSI escape codes on Windows (no-op on other platforms)
os.system("")

def _overwrite(lines: list[str], prev_line_count: int) -> int:
    """
    Overwrite the previous frame in place using ANSI cursor movement.

    Instead of clearing the whole screen (which causes a white flash),
    we move the cursor up by however many lines we printed last frame,
    then reprint. The terminal updates each character in place — no blank
    intermediate state, no flash.

    \033[{n}A  — move cursor up n lines
    \033[J     — erase from cursor to end of screen
    """
    if prev_line_count > 0:
        sys.stdout.write(f"\033[{prev_line_count}A\033[J")
    text = "\n".join(lines)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return text.count("\n") + 1  # return line count for next call


def watch(seed: int, delay: float, greedy: bool, checkpoint: str):
    env   = TowerEnv(num_floors=5, floor_width=5, floor_height=4, max_steps=400)
    agent = DQNAgent(obs_size=OBS_SIZE, n_actions=N_ACTIONS)

    if not os.path.exists(checkpoint):
        print(f"No checkpoint found at {checkpoint}. Train first with: python main.py")
        return

    agent.load(checkpoint)
    agent.epsilon = 0.0 if greedy else 0.05

    obs, _       = env.reset(seed=seed)
    total_reward = 0.0
    step         = 0
    history      = []
    prev_lines   = 0

    print(f"Loaded: {checkpoint}  |  Seed: {seed}  |  Greedy: {greedy}")
    print("Press Ctrl+C to stop early.\n")
    time.sleep(1.0)

    try:
        while True:
            mask   = env.valid_action_mask()
            action = agent.select_action(obs, mask)
            name   = action_name(action)
            top_q  = top_q_values(agent, obs, mask)

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step         += 1
            history.append((name, reward))

            frame = [
                "=" * 56,
                f"  THE ALCHEMIST'S TOWER  --  seed {seed}",
                "=" * 56,
                env.render(),
                "",
                "Recent actions:",
            ]
            for past_name, past_r in history[-5:]:
                marker = ">>> " if (past_name, past_r) == history[-1] else "    "
                frame.append(f"  {marker}{past_name:40s}  r={past_r:+.2f}")
            frame += [
                "",
                f"Top Q: {top_q}",
                "",
                f"Step: {step:4d}  |  Total reward: {total_reward:+7.1f}  |  Floor: {info['floor']+1}/5",
            ]

            prev_lines = _overwrite(frame, prev_lines)
            obs = next_obs
            time.sleep(delay)

            if terminated or truncated:
                from collections import Counter
                counts = Counter(n for n, _ in history)
                end_frame = [
                    "=" * 56,
                    f"  EPISODE OVER -- {'WON!' if env._won else 'Died' if env._hp <= 0 else 'Timed out'}",
                    "=" * 56,
                    env.render(),
                    "",
                    f"Steps:          {step}",
                    f"Floors reached: {info['floor']+1} / 5",
                    f"Total reward:   {total_reward:+.1f}",
                    "",
                    "Action breakdown:",
                ]
                for aname, count in counts.most_common(8):
                    bar = "#" * (count // 2)
                    end_frame.append(f"  {aname:40s} {count:3d}x  {bar}")
                _overwrite(end_frame, prev_lines)
                break

    except KeyboardInterrupt:
        print(f"\nStopped at step {step}. Total reward: {total_reward:+.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int,   default=42,                    help="Map seed")
    parser.add_argument("--delay",      type=float, default=0.15,                  help="Seconds between steps")
    parser.add_argument("--greedy",     action="store_true",                        help="Pure greedy (epsilon=0)")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/best.pt", help="Model checkpoint path")
    args = parser.parse_args()

    watch(args.seed, args.delay, args.greedy, args.checkpoint)
