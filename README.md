# The Alchemist's Tower

A procedurally generated roguelike tower where a Deep Q-Network (DQN) agent learns to collect ingredients, craft potions, and climb as high as possible.

## What's happening

The agent navigates a grid of rooms across multiple tower floors. Each floor is randomly generated with:
- **Ingredients** scattered around (7 types, with rarity weights)
- **Locked doors** that need a Key Essence potion to open
- **Barriers** that need an Explosive potion to destroy
- **Enemies** that scale in HP and attack with floor depth
- **Stairs** placed as far from the start as possible (via BFS)

The agent must combine ingredients into potions using hardcoded recipes, and use those potions strategically to progress upward.

```
+-+-+-+-+-+
|@|. EL. i|       @ = agent        E = enemy
+ + + +-+ +       ^ = stairs up    i = ingredient
|i|^|. i .|       L = locked door  B = barrier
+ +-+-+-+ +
|i i . i .|
+-+-+-+-+-+
```

## Alchemy recipes

| Ingredients | Potion | Effect |
|---|---|---|
| Red Herb + Blue Crystal | Healing | Restores 6 HP |
| Red Herb + Fire Flower | Strength | Double attack (one hit) |
| Blue Crystal + Stone Dust | Key Essence | Opens a locked door |
| Blue Crystal + Shadow Moss | Invisibility | Skip an enemy encounter |
| Fire Flower + Stone Dust | Explosive | Destroys a barrier |
| Golden Root + Void Shard | Floor Skip | Jump up one floor |
| Red Herb + Shadow Moss | Antidote | Cures poison |

## Architecture

```
src/
├── game/
│   ├── items.py      # Ingredient & Potion enums (IntEnum for array indexing)
│   ├── alchemy.py    # Hardcoded recipes as frozenset dict (order-invariant)
│   ├── room.py       # Room dataclass: passages, enemies, ingredients
│   └── floor.py      # Procedural generation: maze → locks → enemies → stairs
├── env/
│   └── tower_env.py  # Gymnasium wrapper: 20 actions, 40-dim observation
└── agent/
    ├── dqn.py        # QNetwork, ReplayBuffer, DQNAgent
    └── train.py      # Curriculum training loop + reward plot
```

## How the RL works

**Observation (40 floats, all in [0, 1]):**
- Agent HP, poison status, position, current floor
- Passage states in all 4 directions (open / locked / barrier)
- Ingredient counts in current room
- Ingredient counts in inventory
- Potion counts in inventory

**Action space (Discrete 20):**
- Move N / E / S / W
- Pick up ingredients
- Attack enemy
- Craft one of 7 potions
- Use one of 7 potions

**Reward shaping:**
- `+10` per floor cleared, `+50` for reaching the top
- `+1` per ingredient collected, `+2` per potion crafted
- `+0.5` for visiting a new room (exploration bonus)
- `-0.1` per step (time penalty — encourages efficiency)
- `-10` on death

**DQN details:**
- Network: `40 → 128 → 128 → 20` with ReLU
- Replay buffer: 20,000 experiences, random batch sampling
- Target network: hard-synced every 200 learn steps
- Epsilon-greedy: 1.0 → 0.05 with exponential decay
- Gradient clipping at norm 10

**Curriculum learning:**

| Stage | Floors | Grid | Advances when |
|---|---|---|---|
| 0 | 1 | 4×3 | mean reward ≥ -10 |
| 1 | 2 | 4×3 | mean reward ≥ 0 |
| 2 | 3 | 5×4 | mean reward ≥ 20 |
| 3 | 5 | 5×4 | (final stage) |

## Setup

```bash
pip install -r requirements.txt
```

## Run training

```bash
python main.py
```

Saves `checkpoints/best.pt` and `checkpoints/training_curve.png` when done.

## Why this project is interesting for RL

- **Inventory management** is a genuinely hard combinatorial subproblem — with 7 ingredient types there are 21 possible pairs, only 7 produce potions
- **Procedural generation** tests generalisation — did the agent learn the game, or memorise a map?
- **Reward shaping** is natural and multi-level
- **Curriculum learning** lets you measure progress incrementally
- The **state encoder** is a multi-hot vector — no LLM required
