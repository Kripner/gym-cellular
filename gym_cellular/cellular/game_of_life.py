import abc
import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.envs.registration import register

from gym_cellular.cellular.automaton import CellularAutomaton


class GameOfLife(CellularAutomaton):
    """
    Concrete implementation of Conway's Game of Life.
    """
    def __init__(self, width: int, height: int, init_random: bool = True):
        super().__init__(width, height)
        if init_random:
            # Random initialization: 0 or 1
            self.state = np.random.randint(2, size=(height, width), dtype=np.uint8)
        else:
            self.state = np.zeros((height, width), dtype=np.uint8)

    def step(self):
        # Compute neighbors
        padded = np.pad(self.state, pad_width=1, mode='wrap')
        new_state = np.zeros_like(self.state)
        for y in range(self.height):
            for x in range(self.width):
                # Sum of 8 neighbors
                total = (
                    padded[y  , x  ] + padded[y  , x+1] + padded[y  , x+2]
                  + padded[y+1, x  ]               + padded[y+1, x+2]
                  + padded[y+2, x  ] + padded[y+2, x+1] + padded[y+2, x+2]
                )
                if self.state[y, x] == 1:
                    # Live cell
                    if total == 2 or total == 3:
                        new_state[y, x] = 1
                    else:
                        new_state[y, x] = 0
                else:
                    # Dead cell
                    if total == 3:
                        new_state[y, x] = 1
                    else:
                        new_state[y, x] = 0
        self.state = new_state

    def possible_states(self) -> list[int]:
       """
       In Game of Life, cells can only be 0 (dead) or 1 (alive).
       """
       return [0, 1]