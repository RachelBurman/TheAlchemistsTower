from __future__ import annotations
import random
from typing import Optional
from .room import Room, Direction, PassageState, OPPOSITE, DELTA, Enemy
from .items import Ingredient, INGREDIENT_RARITY


class Floor:
    """
    A single level of the tower: a W x H grid of Rooms.

    HOW PROCEDURAL GENERATION WORKS HERE
    =====================================
    Step 1 — Carve a perfect maze using the "recursive backtracker" algorithm.
              A perfect maze has exactly ONE path between any two rooms.
              Every room is reachable. No loops.

    Step 2 — Add a few extra passages to create loops.
              Pure mazes are too linear; loops force the agent to make
              choices about which path to explore.

    Step 3 — Lock / barrier some passages based on floor difficulty.

    Step 4 — Scatter ingredients using rarity weights.

    Step 5 — Place enemies, scaling HP/attack with floor number.

    Step 6 — Place the staircase in a room far from the start.

    WHY SEEDED RANDOMNESS?
    ======================
    We pass a `seed` to random.Random() so the same seed always produces
    the same floor. This is essential for RL:
      - Reproducible evaluation: compare agent runs fairly.
      - Curriculum learning: fix easy seeds while training, then randomise.
      - Debugging: replay any generated floor exactly.
    """

    def __init__(self, width: int, height: int, floor_number: int, seed: int):
        self.width        = width
        self.height       = height
        self.floor_number = floor_number
        self.seed         = seed
        self.rng          = random.Random(seed)

        # Build the 2D grid of empty rooms
        self.grid: list[list[Room]] = [
            [Room(x, y) for x in range(width)]
            for y in range(height)
        ]

        self.start_pos: tuple[int, int] = (0, 0)

        self._carve_maze()
        self._add_loops(count=max(1, (width * height) // 6))
        self._lock_passages()
        self._scatter_ingredients()
        self._place_enemies()
        self._place_stairs()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def room_at(self, x: int, y: int) -> Room:
        return self.grid[y][x]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbor(self, x: int, y: int, d: Direction) -> Optional[Room]:
        dx, dy = DELTA[d]
        nx, ny = x + dx, y + dy
        return self.grid[ny][nx] if self.in_bounds(nx, ny) else None

    # ------------------------------------------------------------------
    # Step 1: Recursive backtracker maze
    # ------------------------------------------------------------------

    def _carve_maze(self) -> None:
        """
        The recursive backtracker (also called "depth-first search maze"):

        1. Pick a starting cell, mark it visited.
        2. Push it onto a stack.
        3. While the stack is not empty:
           a. Look at the top cell.
           b. If it has unvisited neighbours, pick one at random,
              remove the wall between them, push the neighbour.
           c. If no unvisited neighbours, pop (backtrack).

        Result: a spanning tree of the grid — every room connected,
        no loops, exactly one path between any two rooms.

        We track "visited" with a local set here, not room.visit_count,
        because room.visit_count tracks agent visits during an episode.
        """
        visited: set[tuple[int, int]] = set()
        stack:   list[tuple[int, int]] = []

        sx, sy = self.start_pos
        visited.add((sx, sy))
        stack.append((sx, sy))

        while stack:
            x, y = stack[-1]
            # Shuffle directions so the maze is different every seed
            directions = list(Direction)
            self.rng.shuffle(directions)

            moved = False
            for d in directions:
                dx, dy = DELTA[d]
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and (nx, ny) not in visited:
                    # Carve a passage both ways (walls are symmetric)
                    self.grid[y][x].passages[d]              = PassageState.OPEN
                    self.grid[ny][nx].passages[OPPOSITE[d]]  = PassageState.OPEN
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                    moved = True
                    break

            if not moved:
                stack.pop()  # dead end — backtrack

    # ------------------------------------------------------------------
    # Step 2: Add loops
    # ------------------------------------------------------------------

    def _add_loops(self, count: int) -> None:
        """
        Pick `count` random solid walls and open them.
        This turns the perfect maze into a "braided" maze with multiple
        paths — which is more interesting and realistic.
        """
        attempts = 0
        added    = 0
        while added < count and attempts < count * 10:
            attempts += 1
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            d = self.rng.choice(list(Direction))
            nb = self.neighbor(x, y, d)
            if nb and self.grid[y][x].passages[d] == PassageState.WALL:
                self.grid[y][x].passages[d]              = PassageState.OPEN
                nb.passages[OPPOSITE[d]]                 = PassageState.OPEN
                added += 1

    # ------------------------------------------------------------------
    # Step 3: Lock some passages
    # ------------------------------------------------------------------

    def _lock_passages(self) -> None:
        """
        Convert some OPEN passages to LOCKED or BARRIER.
        We never lock the passages touching the start room — the agent
        would be immediately stuck.

        Difficulty scales with floor_number: more locks on higher floors.
        """
        sx, sy       = self.start_pos
        n_locked     = min(self.floor_number, self.width * self.height // 3)
        n_barriers   = min(self.floor_number // 2, 2)

        for room_list, passage_type, count in [
            (self._all_open_passages(exclude_start=True), PassageState.LOCKED,  n_locked),
            (self._all_open_passages(exclude_start=True), PassageState.BARRIER, n_barriers),
        ]:
            chosen = self.rng.sample(room_list, min(count, len(room_list)))
            for x, y, d in chosen:
                nb = self.neighbor(x, y, d)
                if nb:
                    self.grid[y][x].passages[d]    = passage_type
                    nb.passages[OPPOSITE[d]]        = passage_type

    def _all_open_passages(self, exclude_start: bool = False) -> list[tuple[int, int, Direction]]:
        sx, sy = self.start_pos
        result = []
        for y in range(self.height):
            for x in range(self.width):
                if exclude_start and (x, y) == (sx, sy):
                    continue
                for d in (Direction.EAST, Direction.SOUTH):  # avoid counting each passage twice
                    if self.grid[y][x].passages[d] == PassageState.OPEN:
                        result.append((x, y, d))
        return result

    # ------------------------------------------------------------------
    # Step 4: Scatter ingredients
    # ------------------------------------------------------------------

    def _scatter_ingredients(self) -> None:
        """
        Weighted random sampling: high-rarity-weight ingredients appear
        more often. We also scale total ingredients with floor size.

        WHY weighted sampling matters for RL:
        The agent must learn that Void Shard is rare and worth keeping,
        while Red Herb can be found again soon.
        """
        ingredients = list(INGREDIENT_RARITY.keys())
        weights     = list(INGREDIENT_RARITY.values())
        total       = self.width * self.height  # one ingredient per room on average

        for _ in range(total):
            ing  = self.rng.choices(ingredients, weights=weights, k=1)[0]
            x    = self.rng.randrange(self.width)
            y    = self.rng.randrange(self.height)
            if not self.grid[y][x].has_stairs:
                self.grid[y][x].ingredients.append(ing)

    # ------------------------------------------------------------------
    # Step 5: Place enemies
    # ------------------------------------------------------------------

    def _place_enemies(self) -> None:
        """
        Enemy difficulty scales with floor_number.
        We skip the start room — dying immediately is unfun and teaches nothing.
        Some enemies are poisonous (the agent needs to craft Antidote).
        """
        sx, sy      = self.start_pos
        n_enemies   = max(1, self.floor_number)
        base_hp     = 5 + self.floor_number * 3
        base_atk    = 1 + self.floor_number

        rooms = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != (sx, sy)
        ]
        self.rng.shuffle(rooms)

        for x, y in rooms[:n_enemies]:
            hp        = self.rng.randint(base_hp - 2, base_hp + 2)
            atk       = self.rng.randint(base_atk - 1, base_atk + 1)
            poisonous = self.rng.random() < 0.2 * (self.floor_number / 5)
            self.grid[y][x].enemies.append(Enemy(hp=hp, attack=atk, poisonous=poisonous))

    # ------------------------------------------------------------------
    # Step 6: Place stairs
    # ------------------------------------------------------------------

    def _place_stairs(self) -> None:
        """
        BFS from start to find the room with the greatest distance.
        Putting the stairs there maximises the exploration the agent must do.
        """
        sx, sy   = self.start_pos
        dist     = {(sx, sy): 0}
        queue    = [(sx, sy)]

        while queue:
            x, y  = queue.pop(0)
            for d in Direction:
                if self.grid[y][x].passages[d] in (PassageState.OPEN,
                                                    PassageState.LOCKED,
                                                    PassageState.BARRIER):
                    dx, dy = DELTA[d]
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in dist:
                        dist[(nx, ny)] = dist[(x, y)] + 1
                        queue.append((nx, ny))

        # Farthest reachable room gets the staircase
        fx, fy = max(dist, key=dist.get)
        self.grid[fy][fx].has_stairs = True

    # ------------------------------------------------------------------
    # ASCII rendering
    # ------------------------------------------------------------------

    def render(self, agent_pos: Optional[tuple[int, int]] = None) -> str:
        """
        Draw the floor as ASCII art. Each room is a 3x3 block of characters.

        Legend:
          @  agent position
          ^  stairs up
          E  enemy present
          i  ingredient present
          .  empty room
          |  vertical wall / locked (L) / barrier (B)
          -  horizontal wall
        """
        lines = []
        for y in range(self.height):
            # Top border row of this rank of rooms
            top_row = []
            for x in range(self.width):
                top_row.append("+")
                n_state = self.grid[y][x].passages[Direction.NORTH]
                if n_state == PassageState.OPEN:
                    top_row.append(" ")
                elif n_state == PassageState.LOCKED:
                    top_row.append("L")
                elif n_state == PassageState.BARRIER:
                    top_row.append("B")
                else:
                    top_row.append("-")
            top_row.append("+")
            lines.append("".join(top_row))

            # Middle row (room content + east walls)
            mid_row = []
            for x in range(self.width):
                w_state = self.grid[y][x].passages[Direction.WEST]
                if w_state == PassageState.OPEN:
                    mid_row.append(" ")
                elif w_state == PassageState.LOCKED:
                    mid_row.append("L")
                elif w_state == PassageState.BARRIER:
                    mid_row.append("B")
                else:
                    mid_row.append("|")

                if agent_pos and (x, y) == agent_pos:
                    mid_row.append("@")
                else:
                    mid_row.append(self.grid[y][x].ascii_char())

            # Final east wall of the row
            e_state = self.grid[y][self.width - 1].passages[Direction.EAST]
            mid_row.append("|" if e_state != PassageState.OPEN else " ")
            lines.append("".join(mid_row))

        # Bottom border of entire floor
        bottom = []
        for x in range(self.width):
            s_state = self.grid[self.height - 1][x].passages[Direction.SOUTH]
            bottom.append("+")
            bottom.append("-" if s_state != PassageState.OPEN else " ")
        bottom.append("+")
        lines.append("".join(bottom))

        return "\n".join(lines)
