from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from game.floor import Floor
from game.room import Direction, PassageState, DELTA, OPPOSITE
from game.items import Ingredient, Potion
from game.alchemy import RECIPES, craft, all_craftable

# -----------------------------------------------------------------------
# HOW GYMNASIUM ENVIRONMENTS WORK
# -----------------------------------------------------------------------
# Every RL environment follows this contract:
#
#   obs, info          = env.reset()         # start a new episode
#   obs, reward, terminated, truncated, info = env.step(action)
#
# The agent calls reset() once, then step() in a loop until terminated
# or truncated is True. That loop is one "episode."
#
# observation_space and action_space are Gymnasium Space objects that
# describe the shape/dtype/bounds of observations and valid actions.
# The RL algorithm reads these to know what size neural network to build.
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# ACTION SPACE  (Discrete — one integer per action)
# -----------------------------------------------------------------------
# 0-3  : Move NORTH / EAST / SOUTH / WEST
# 4    : Pick up all ingredients in current room
# 5    : Attack enemy in current room
# 6-12 : Craft a potion (one action per recipe, 7 recipes)
# 13-19: Use a potion  (one action per potion type, 7 types)
#
# Total: 20 discrete actions
#
# WHY discrete and not continuous?
#   Our actions are fundamentally categorical — there's no meaningful
#   "halfway between move north and pick up." Discrete(20) is the
#   natural fit, and DQN is designed for discrete action spaces.
# -----------------------------------------------------------------------

N_MOVE_ACTIONS  = 4
ACTION_PICKUP   = 4
ACTION_ATTACK   = 5
ACTION_CRAFT_BASE = 6
N_RECIPES       = 7   # must match len(RECIPES)
ACTION_USE_BASE = ACTION_CRAFT_BASE + N_RECIPES   # 13
N_POTIONS       = 7   # must match len(Potion)
N_ACTIONS       = ACTION_USE_BASE + N_POTIONS     # 20

DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

# Stable ordered list of recipes so action index -> recipe is deterministic
RECIPE_LIST: list[tuple[Ingredient, Ingredient, Potion]] = [
    (a, b, p)
    for key, p in RECIPES.items()
    for a, b in [tuple(key)]
]

# -----------------------------------------------------------------------
# OBSERVATION SPACE  (40-dimensional float32 vector, all values in [0,1])
# -----------------------------------------------------------------------
# [0]     hp / max_hp
# [1]     poisoned (0 or 1)
# [2]     agent_x / (width-1)
# [3]     agent_y / (height-1)
# [4]     current_floor_idx / num_floors
# [5]     room has stairs
# [6]     room has enemy
# [7-18]  passages N/E/S/W × (is_open, is_locked, is_barrier) — 3 bits each
# [19-25] count of each ingredient in current room  / 3  (capped)
# [26-32] count of each ingredient in inventory     / 5  (capped)
# [33-39] count of each potion in inventory         / 3  (capped)
#
# WHY a flat vector?
#   Neural networks take tensors. We need to squash all game state into
#   one fixed-size array. The multi-hot / count encoding is efficient:
#   7 ingredient types = 7 numbers, not a variable-length list.
#
# WHY normalise to [0,1]?
#   Most NN activation functions (ReLU, tanh, sigmoid) behave well when
#   inputs are small. Raw HP=20 mixed with binary flags would confuse
#   gradient descent — large values dominate early gradients.
# -----------------------------------------------------------------------

OBS_SIZE = 40


class TowerEnv(gym.Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        num_floors:   int = 5,
        floor_width:  int = 5,
        floor_height: int = 4,
        max_steps:    int = 500,
        render_mode:  Optional[str] = None,
    ):
        super().__init__()
        self.num_floors   = num_floors
        self.floor_width  = floor_width
        self.floor_height = floor_height
        self.max_steps    = max_steps
        self.render_mode  = render_mode

        # Agent constants
        self.max_hp  = 20
        self.attack  = 3

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # State (populated by reset)
        self._floor:      Optional[Floor] = None
        self._floor_idx:  int = 0
        self._agent_pos:  tuple[int, int] = (0, 0)
        self._hp:         int = self.max_hp
        self._poisoned:   bool = False
        self._inv_ing:    dict[Ingredient, int] = {}
        self._inv_pot:    dict[Potion, int] = {}
        self._steps:      int = 0
        self._episode_seed: Optional[int] = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    # Called at the start of each episode. Must return (obs, info).
    # The `seed` parameter lets the caller control randomness — important
    # for reproducible evaluation runs.
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_seed = seed if seed is not None else self.np_random.integers(0, 2**31)

        self._floor_idx = 0
        self._hp        = self.max_hp
        self._poisoned  = False
        self._inv_ing   = {i: 0 for i in Ingredient}
        self._inv_pot   = {p: 0 for p in Potion}
        self._steps     = 0
        self._won       = False

        self._load_floor(self._floor_idx)
        return self._observe(), {}

    def _load_floor(self, floor_idx: int) -> None:
        """Generate the floor for floor_idx. Seed is offset so each floor differs."""
        seed = int(self._episode_seed) + floor_idx * 1000
        self._floor     = Floor(self.floor_width, self.floor_height, floor_idx + 1, seed)
        self._agent_pos = self._floor.start_pos
        self._floor.room_at(*self._agent_pos).visited = True

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    # The core game loop. Returns (obs, reward, terminated, truncated, info).
    #
    #   terminated = episode ended because of game logic (died / won)
    #   truncated  = episode ended because we hit max_steps
    #
    # Separating these two matters: a truncated episode shouldn't be
    # treated as a "loss" by the agent — it just ran out of time.
    # ------------------------------------------------------------------

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._steps += 1
        reward = -0.01  # small time penalty every step — encourages efficiency

        if action < N_MOVE_ACTIONS:
            reward += self._act_move(DIRECTIONS[action])

        elif action == ACTION_PICKUP:
            reward += self._act_pickup()

        elif action == ACTION_ATTACK:
            reward += self._act_attack()

        elif ACTION_CRAFT_BASE <= action < ACTION_USE_BASE:
            recipe_idx = action - ACTION_CRAFT_BASE
            reward += self._act_craft(recipe_idx)

        else:
            potion_idx = action - ACTION_USE_BASE
            potion     = list(Potion)[potion_idx]
            reward += self._act_use_potion(potion)

        # Poison tick
        if self._poisoned:
            self._hp -= 1

        terminated = self._hp <= 0 or self._won
        truncated  = self._steps >= self.max_steps

        if self._hp <= 0 and not self._won:
            reward += -10.0  # death penalty

        obs  = self._observe()
        info = {
            "floor":   self._floor_idx,
            "hp":      self._hp,
            "steps":   self._steps,
            "pos":     self._agent_pos,
        }

        if self.render_mode == "ansi":
            print(self.render())

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _act_move(self, direction: Direction) -> float:
        """
        Try to move one room in `direction`.
        - OPEN passage: move freely, +0.5 if room is new (exploration bonus)
        - LOCKED: check inventory for Key Essence, consume it to open
        - BARRIER: check for Explosive, consume to destroy
        - WALL / no passage: invalid move, -0.5 penalty

        WHY reward new rooms?
          Without an exploration bonus, the agent learns to stand still
          (no time penalty beats -0.1/step from wandering). The +0.5
          for new rooms pulls it to explore without over-weighting it.
        """
        x, y = self._agent_pos
        room = self._floor.room_at(x, y)
        state = room.passages[direction]

        dx, dy = DELTA[direction]
        nx, ny = x + dx, y + dy

        if not self._floor.in_bounds(nx, ny) or state == PassageState.WALL:
            return 0.0  # masked out — shouldn't be selected

        if state == PassageState.LOCKED:
            if self._inv_pot[Potion.KEY_ESSENCE] > 0:
                self._inv_pot[Potion.KEY_ESSENCE] -= 1
                room.open_passage(direction)
                self._floor.room_at(nx, ny).open_passage(OPPOSITE[direction])
            else:
                return 0.0  # masked out

        if state == PassageState.BARRIER:
            if self._inv_pot[Potion.EXPLOSIVE] > 0:
                self._inv_pot[Potion.EXPLOSIVE] -= 1
                room.open_passage(direction)
                self._floor.room_at(nx, ny).open_passage(OPPOSITE[direction])
            else:
                return 0.0  # masked out

        # Move
        self._agent_pos = (nx, ny)
        new_room  = self._floor.room_at(nx, ny)
        is_new    = not new_room.visited
        new_room.visited = True

        reward = 0.5 if is_new else 0.0  # exploration bonus for first visit

        if new_room.has_stairs:
            reward += self._climb_stairs()

        return reward

    def _climb_stairs(self) -> float:
        """Advance to the next floor, or win if at the top."""
        self._floor_idx += 1
        if self._floor_idx >= self.num_floors:
            self._won = True
            return +50.0   # win!
        self._load_floor(self._floor_idx)
        return +10.0  # floor cleared

    def _act_pickup(self) -> float:
        """Collect all ingredients in the current room."""
        x, y = self._agent_pos
        room = self._floor.room_at(x, y)
        if not room.ingredients:
            return 0.0  # masked out
        reward = len(room.ingredients) * 1.0  # +1 per ingredient collected
        for ing in room.ingredients:
            self._inv_ing[ing] += 1
        room.ingredients.clear()
        return reward

    def _act_attack(self) -> float:
        """Fight the first living enemy in the room. Agent and enemy trade hits."""
        x, y  = self._agent_pos
        room  = self._floor.room_at(x, y)
        enemy = next((e for e in room.enemies if e.is_alive), None)
        if enemy is None:
            return 0.0  # masked out

        # Strength potion doubles attack for one hit
        atk = self.attack
        if self._inv_pot[Potion.STRENGTH] > 0:
            self._inv_pot[Potion.STRENGTH] -= 1
            atk *= 2

        enemy.take_damage(atk)
        reward = 0.0

        if not enemy.is_alive:
            room.enemies.remove(enemy)
            reward += 2.0  # enemy defeated
        else:
            self._hp -= enemy.attack
            if enemy.poisonous and not self._poisoned:
                self._poisoned = True

        return reward

    def _act_craft(self, recipe_idx: int) -> float:
        """
        Attempt to craft using the recipe at RECIPE_LIST[recipe_idx].
        Deducts both ingredients from inventory on success.

        WHY one action per recipe (not one generic "craft")?
          The agent must choose WHICH potion to make. That choice
          is the combinatorial subproblem — craft Key Essence now
          to open the locked door, or craft Healing first because
          HP is low? Separate actions let the agent learn this.
        """
        if recipe_idx >= len(RECIPE_LIST):
            return 0.0
        a, b, potion = RECIPE_LIST[recipe_idx]
        if self._inv_ing[a] < 1 or self._inv_ing[b] < 1:
            return 0.0  # masked out
        self._inv_ing[a] -= 1
        self._inv_ing[b] -= 1
        self._inv_pot[potion] += 1
        return +2.0  # crafting reward

    def _act_use_potion(self, potion: Potion) -> float:
        """Use a potion from inventory. Each type has a different effect."""
        if self._inv_pot[potion] < 1:
            return 0.0  # masked out

        self._inv_pot[potion] -= 1

        if potion == Potion.HEALING:
            healed        = min(self.max_hp - self._hp, 6)
            self._hp     += healed
            return healed * 0.3   # reward scales with HP actually recovered

        if potion == Potion.ANTIDOTE:
            if self._poisoned:
                self._poisoned = False
                return +1.0
            return 0.0  # masked out — not poisoned

        if potion == Potion.FLOOR_SKIP:
            return self._climb_stairs()

        # STRENGTH / INVISIBILITY / EXPLOSIVE / KEY_ESSENCE are consumed
        # when used contextually (attack / move), not here directly.
        # Using them here is a no-op but we still consumed the potion.
        return 0.0  # masked out

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------
    # WHY action masking?
    #   Without it, the agent wastes steps on impossible actions (craft a
    #   potion it has no ingredients for, move through a wall) and learns
    #   to avoid them via negative reward — which distorts the Q-values.
    #   With masking, invalid Q-values are set to -inf before the argmax,
    #   so the agent never selects them. Every step is a meaningful choice.

    def valid_action_mask(self) -> np.ndarray:
        """Return a boolean array of length N_ACTIONS: True = action is valid now."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        x, y = self._agent_pos
        room = self._floor.room_at(x, y)

        # Movement: valid if passage is open, or locked/barrier with the right potion
        for i, d in enumerate(DIRECTIONS):
            state = room.passages[d]
            dx, dy = DELTA[d]
            if not self._floor.in_bounds(x + dx, y + dy):
                continue
            if state == PassageState.OPEN:
                mask[i] = True
            elif state == PassageState.LOCKED and self._inv_pot[Potion.KEY_ESSENCE] > 0:
                mask[i] = True
            elif state == PassageState.BARRIER and self._inv_pot[Potion.EXPLOSIVE] > 0:
                mask[i] = True

        # Pickup: valid if room has ingredients
        mask[ACTION_PICKUP] = bool(room.ingredients)

        # Attack: valid if room has a living enemy
        mask[ACTION_ATTACK] = any(e.is_alive for e in room.enemies)

        # Craft: valid if we have both ingredients for that recipe
        for i, (a, b, _) in enumerate(RECIPE_LIST):
            mask[ACTION_CRAFT_BASE + i] = (
                self._inv_ing[a] >= 1 and self._inv_ing[b] >= 1
            )

        # Use potion: valid if we have it (and it's contextually useful)
        for i, potion in enumerate(Potion):
            if self._inv_pot[potion] < 1:
                continue
            if potion == Potion.HEALING:
                mask[ACTION_USE_BASE + i] = self._hp < self.max_hp
            elif potion == Potion.ANTIDOTE:
                mask[ACTION_USE_BASE + i] = self._poisoned
            elif potion == Potion.FLOOR_SKIP:
                mask[ACTION_USE_BASE + i] = True
            # STRENGTH, INVISIBILITY used contextually via attack/move — skip here

        # Safety: always allow at least one action (move to least-bad direction)
        if not mask.any():
            for i in range(N_MOVE_ACTIONS):
                dx, dy = DELTA[DIRECTIONS[i]]
                if self._floor.in_bounds(x + dx, y + dy):
                    mask[i] = True
                    break

        return mask

    # ------------------------------------------------------------------
    # Observation encoder
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """
        Encode the full game state as a 40-dimensional float32 vector.

        This is called after every step and after reset — it's the only
        information the agent ever sees. Everything the agent needs to
        make good decisions must be in here.
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        x, y = self._agent_pos
        room = self._floor.room_at(x, y)

        obs[0] = self._hp / self.max_hp
        obs[1] = float(self._poisoned)
        obs[2] = x / max(self.floor_width  - 1, 1)
        obs[3] = y / max(self.floor_height - 1, 1)
        obs[4] = self._floor_idx / max(self.num_floors - 1, 1)
        obs[5] = float(room.has_stairs)
        obs[6] = float(any(e.is_alive for e in room.enemies))

        # Passage encoding: 3 bits per direction = 12 values at [7..18]
        for i, d in enumerate(DIRECTIONS):
            base  = 7 + i * 3
            state = room.passages[d]
            obs[base + 0] = float(state == PassageState.OPEN)
            obs[base + 1] = float(state == PassageState.LOCKED)
            obs[base + 2] = float(state == PassageState.BARRIER)

        # Room ingredient counts (normalised by 3)
        for ing in Ingredient:
            obs[19 + ing - 1] = min(room.ingredients.count(ing), 3) / 3.0

        # Inventory ingredient counts (normalised by 5)
        for ing in Ingredient:
            obs[26 + ing - 1] = min(self._inv_ing[ing], 5) / 5.0

        # Inventory potion counts (normalised by 3)
        for pot in Potion:
            obs[33 + pot - 1] = min(self._inv_pot[pot], 3) / 3.0

        return obs

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> str:
        if self._floor is None:
            return "No floor loaded."
        header = (
            f"Floor {self._floor_idx + 1}/{self.num_floors}  "
            f"HP: {self._hp}/{self.max_hp}  "
            f"{'[POISONED] ' if self._poisoned else ''}"
            f"Step: {self._steps}"
        )
        inv_i = "  ".join(
            f"{i.name[:3]}:{self._inv_ing[i]}"
            for i in Ingredient if self._inv_ing[i] > 0
        )
        inv_p = "  ".join(
            f"{p.name[:3]}:{self._inv_pot[p]}"
            for p in Potion if self._inv_pot[p] > 0
        )
        inv_line = f"Ingredients: {inv_i or 'none'}   Potions: {inv_p or 'none'}"
        return "\n".join([header, inv_line, self._floor.render(self._agent_pos)])
