from __future__ import annotations
from typing import Optional
from .items import Ingredient, Potion

# The recipe book is a plain dict with frozenset keys.
#
# WHY frozenset?
#   A set is unordered — frozenset({A, B}) == frozenset({B, A}).
#   That means "Red Herb + Blue Crystal" and "Blue Crystal + Red Herb"
#   are the same recipe automatically. No duplicate entries needed.
#   It's also hashable (unlike a regular set), so it works as a dict key.
#
# WHY hardcode this instead of using an LLM?
#   The agent's job is to DISCOVER these rules through exploration and
#   reward signals. If the rules changed every episode, we'd need an LLM
#   to reason about them. Fixed rules let us measure whether the agent
#   has actually learned the crafting system.

RECIPES: dict[frozenset, Potion] = {
    frozenset({Ingredient.RED_HERB,    Ingredient.BLUE_CRYSTAL}): Potion.HEALING,
    frozenset({Ingredient.RED_HERB,    Ingredient.FIRE_FLOWER}):  Potion.STRENGTH,
    frozenset({Ingredient.BLUE_CRYSTAL,Ingredient.STONE_DUST}):   Potion.KEY_ESSENCE,
    frozenset({Ingredient.BLUE_CRYSTAL,Ingredient.SHADOW_MOSS}):  Potion.INVISIBILITY,
    frozenset({Ingredient.FIRE_FLOWER, Ingredient.STONE_DUST}):   Potion.EXPLOSIVE,
    frozenset({Ingredient.GOLDEN_ROOT, Ingredient.VOID_SHARD}):   Potion.FLOOR_SKIP,
    frozenset({Ingredient.RED_HERB,    Ingredient.SHADOW_MOSS}):  Potion.ANTIDOTE,
}


def craft(a: Ingredient, b: Ingredient) -> Optional[Potion]:
    """Return the potion produced by combining two ingredients, or None."""
    return RECIPES.get(frozenset({a, b}))


def all_craftable(inventory: list[Ingredient]) -> list[tuple[Ingredient, Ingredient, Potion]]:
    """
    Given a list of ingredients the agent is carrying, return every
    valid (a, b, potion) triple it could craft right now.

    This is the combinatorial subproblem: with N ingredients there are
    N*(N-1)/2 possible pairs. The agent must learn which pairs are worth
    crafting given its current situation (locked door ahead? craft Key Essence).
    """
    results = []
    seen = set()
    for i, a in enumerate(inventory):
        for b in inventory[i + 1:]:
            key = frozenset({a, b})
            if key in seen:
                continue
            seen.add(key)
            potion = RECIPES.get(key)
            if potion is not None:
                results.append((a, b, potion))
    return results


def recipe_table() -> str:
    """Pretty-print the recipe book — useful for debugging."""
    lines = ["ALCHEMY RECIPES", "=" * 40]
    for ingredients, potion in RECIPES.items():
        a, b = tuple(ingredients)
        lines.append(f"  {a.name} + {b.name} -> {potion.name}")
    return "\n".join(lines)
