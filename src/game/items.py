from enum import IntEnum, auto


class Ingredient(IntEnum):
    """
    IntEnum gives each ingredient a unique integer automatically (auto()).
    We use IntEnum (not just Enum) so ingredients can be used directly
    as array indices when building the multi-hot state vector later.
    """
    RED_HERB    = auto()   # 1 — common, found on low floors
    BLUE_CRYSTAL = auto()  # 2 — common
    FIRE_FLOWER  = auto()  # 3 — uncommon
    STONE_DUST   = auto()  # 4 — common
    SHADOW_MOSS  = auto()  # 5 — uncommon
    GOLDEN_ROOT  = auto()  # 6 — rare
    VOID_SHARD   = auto()  # 7 — rare, high floors only


class Potion(IntEnum):
    """
    Potions are the agent's tools for solving problems.
    Each one solves a different obstacle in the tower.
    """
    HEALING      = auto()   # Restore HP — fight enemies without dying
    STRENGTH     = auto()   # +attack for one combat — kill tough enemies
    KEY_ESSENCE  = auto()   # Unlocks locked doors — critical for progression
    INVISIBILITY = auto()   # Skip an enemy encounter entirely
    EXPLOSIVE    = auto()   # Destroy a barrier blocking a room
    FLOOR_SKIP   = auto()   # Jump up one floor instantly
    ANTIDOTE     = auto()   # Cure poison (some enemies inflict it)


# Human-readable names for rendering
INGREDIENT_NAMES = {i: i.name.replace("_", " ").title() for i in Ingredient}
POTION_NAMES     = {p: p.name.replace("_", " ").title() for p in Potion}

# Rarity weights — used by procedural generation to decide spawn frequency.
# Higher number = more common. Floor generator will scale these by floor depth.
INGREDIENT_RARITY = {
    Ingredient.RED_HERB:     10,
    Ingredient.BLUE_CRYSTAL: 10,
    Ingredient.STONE_DUST:   8,
    Ingredient.FIRE_FLOWER:  5,
    Ingredient.SHADOW_MOSS:  5,
    Ingredient.GOLDEN_ROOT:  2,
    Ingredient.VOID_SHARD:   1,
}
