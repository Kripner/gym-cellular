import numpy as np
import pygame
from gymnasium import spaces

from gym_cellular.cellular.game_of_life import GameOfLife
from gym_cellular.environment.env import AbstractCellularEnv


class HelicopterEnv(AbstractCellularEnv):
    """
    Environment with a helicopter agent moving on a game-of-life grid.

    Actions: 0-7 correspond to 8 directions:
      0: North
      1: Northeast
      2: East
      3: Southeast
      4: South
      5: Southwest
      6: West
      7: Northwest

    The helicopter moves one cell in the chosen direction each step (with wrap-around).
    """
    def __init__(self, width=50, height=50, render_mode=None, max_steps=1000):
        automaton = GameOfLife(width, height, init_random=True)
        super().__init__(automaton, render_mode)
        # Override action and observation spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.height, self.width),
            dtype=np.uint8
        )
        # Agent position
        self.agent_pos = np.array([self.height // 2, self.width // 2], dtype=int)
        # Step counter
        self.step_count = 0
        self.max_steps = max_steps

    def _reset_agent(self):
        # Place helicopter at center
        self.agent_pos = np.array([self.height // 2, self.width // 2], dtype=int)
        self.step_count = 0

    def _move_agent(self, action: int):
        # Directions: dy, dx for each action
        directions = [(-1, 0),  # North
                      (-1, 1),  # Northeast
                      (0, 1),   # East
                      (1, 1),   # Southeast
                      (1, 0),   # South
                      (1, -1),  # Southwest
                      (0, -1),  # West
                      (-1, -1)] # Northwest
        dy, dx = directions[action]
        new_y = (self.agent_pos[0] + dy) % self.height
        new_x = (self.agent_pos[1] + dx) % self.width
        self.agent_pos = np.array([new_y, new_x], dtype=int)
        self.step_count += 1

    def _get_reward(self) -> float:
        # Example reward: +1 if helicopter lands on a live cell, else 0
        y, x = self.agent_pos
        if self.automaton.state[y, x] == 1:
            return 1.0
        return 0.0

    def _get_terminated(self) -> bool:
        # Episode ends after max_steps
        return self.step_count >= self.max_steps

    def _get_info(self) -> dict:
        return {"agent_pos": tuple(self.agent_pos)}

    def _render_agent(self, surface: pygame.Surface):
        # Draw helicopter as a red square
        y, x = self.agent_pos
        rect = pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(surface, (255, 0, 0), rect)  # red for helicopter
