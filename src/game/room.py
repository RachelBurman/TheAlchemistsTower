from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from .items import Ingredient, Potion


class Direction(IntEnum):
    NORTH = 0
    EAST  = 1
    SOUTH = 2
    WEST  = 3


# These two dicts let us answer "what's the opposite direction?" and
# "if I move north, how does my (x,y) change?" without any if/else chains.
OPPOSITE: dict[Direction, Direction] = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST:  Direction.WEST,
    Direction.WEST:  Direction.EAST,
}

DELTA: dict[Direction, tuple[int, int]] = {
    Direction.NORTH: ( 0, -1),
    Direction.SOUTH: ( 0,  1),
    Direction.EAST:  ( 1,  0),
    Direction.WEST:  (-1,  0),
}


class PassageState(IntEnum):
    """
    The state of a wall between two adjacent rooms.
    WALL    = solid, impassable
    OPEN    = freely walkable
    LOCKED  = needs a Key Essence potion to open
    BARRIER = needs an Explosive potion to destroy
    """
    WALL    = 0
    OPEN    = 1
    LOCKED  = 2
    BARRIER = 3


@dataclass
class Enemy:
    hp:        int
    attack:    int
    poisonous: bool = False

    def take_damage(self, amount: int) -> None:
        self.hp = max(0, self.hp - amount)

    @property
    def is_alive(self) -> bool:
        return self.hp > 0


@dataclass
class Room:
    x: int
    y: int

    # passages maps each direction to what's in that wall.
    # Default: every wall is solid. The floor generator carves openings.
    passages: dict[Direction, PassageState] = field(
        default_factory=lambda: {d: PassageState.WALL for d in Direction}
    )

    ingredients: list[Ingredient] = field(default_factory=list)
    enemies:     list[Enemy]      = field(default_factory=list)
    has_stairs:  bool             = False
    visited:     bool             = False  # has the agent been here?

    def can_enter_from(self, direction: Direction) -> bool:
        """True if the passage from `direction` side is open."""
        return self.passages[direction] == PassageState.OPEN

    def passage_requires(self, direction: Direction) -> Optional[Potion]:
        """Return which potion unlocks this passage, or None if open/wall."""
        state = self.passages[direction]
        if state == PassageState.LOCKED:
            return Potion.KEY_ESSENCE
        if state == PassageState.BARRIER:
            return Potion.EXPLOSIVE
        return None

    def open_passage(self, direction: Direction) -> None:
        self.passages[direction] = PassageState.OPEN

    def ascii_char(self) -> str:
        """Single character for quick ASCII map rendering."""
        if self.has_stairs:
            return "^"
        if self.enemies:
            return "E"
        if self.ingredients:
            return "i"
        return "."
