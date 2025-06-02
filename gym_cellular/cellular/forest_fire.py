import numpy as np
from .automaton import CellularAutomaton

class ForestFire(CellularAutomaton):
    """
    Forest-Fire cellular automaton with the following discrete states:
      0: empty
      1: tree
      2: fire_1
      3: fire_2
      4: fire_3
      5: rock

    Rules (applied synchronously to every cell each step):
      - If state == 0 (empty) AND at least one neighbor is a tree (1),
        become a tree (1).
      - If state == 1 (tree) AND at least one neighbor is any fire (2,3,4),
        become fire_1 (2).
      - If state == 2 (fire_1) → become fire_2 (3).
      - If state == 3 (fire_2) → become fire_3 (4).
      - If state == 4 (fire_3) → become empty (0).
      - If state == 5 (rock) → remain rock forever.
    """
    EMPTY   = 0
    TREE    = 1
    FIRE_1  = 2
    FIRE_2  = 3
    FIRE_3  = 4
    ROCK    = 5

    def __init__(self, width: int, height: int):
        """
        width, height: grid dimensions.
        init_random: if True, populate randomly (with probabilities below).
        p_tree: when random, probability for “tree” per cell; 
                p_empty = (1 – p_tree – p_rock), p_rock fixed at 0.1, p_fire = 0.0 initially.
        """
        super().__init__(width, height)

    def reset(self):
        rng = np.random.default_rng(0)
        self.state = rng.choice([self.TREE, self.ROCK], size=(self.height, self.width), p=[0.9, 0.1])

    def step(self):
        """
        Advance the forest by one time step under the rules described above.
        """
        padded = np.pad(self.state, pad_width=1, mode="constant", constant_values=self.ROCK)
        new_state = np.zeros_like(self.state)

        for y in range(self.height):
            for x in range(self.width):
                current = self.state[y, x]
                # Collect the 4 neighbors from the padded array:
                ny, nx = y + 1, x + 1
                neighbors = np.asarray([
                    padded[ny, nx + 1],
                    padded[ny, nx - 1],
                    padded[ny + 1, nx],
                    padded[ny - 1, nx]
                ])

                if current == self.ROCK:
                    # Rocks stay the same
                    new_state[y, x] = self.ROCK
                elif current == self.EMPTY:
                    # If any neighbor is a TREE, become a TREE
                    if np.any(neighbors == self.TREE):
                        new_state[y, x] = self.TREE
                    else:
                        new_state[y, x] = self.EMPTY
                elif current == self.TREE:
                    if np.any(neighbors == self.FIRE_3):
                        new_state[y, x] = self.FIRE_1
                    else:
                        new_state[y, x] = self.TREE
                elif current == self.FIRE_1:
                    new_state[y, x] = self.FIRE_2
                elif current == self.FIRE_2:
                    new_state[y, x] = self.FIRE_3
                elif current == self.FIRE_3:
                    new_state[y, x] = self.EMPTY
                else:
                    raise ValueError(f"Invalid cell state: {current} on position ({y}, {x})")

        self.state = new_state

    def get_state(self) -> np.ndarray:
        """
        Return a copy of the underlying 2D array (values in {0..5}).
        """
        return self.state.copy()

    def set_state(self, new_state: np.ndarray):
        """
        Replace state with new_state (must match dimensions).
        """
        assert new_state.shape == (self.height, self.width)
        self.state = new_state.copy()

    def possible_states(self) -> list[int]:
        """
        The integer codes for this automaton are exactly [0..5].
        """
        return [
            self.EMPTY,
            self.TREE,
            self.FIRE_1,
            self.FIRE_2,
            self.FIRE_3,
            self.ROCK
        ]