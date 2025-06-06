import numpy as np
import pygame
from gymnasium import spaces

from gym_cellular.cellular.forest_fire import ForestFire
from gym_cellular.environment.env import AbstractCellularEnv


class HelicopterEnv(AbstractCellularEnv):
    """
    Environment with a helicopter agent moving on a game-of-life grid.

    Actions: 0-3 correspond to 4 directions:
      0: North
      1: East
      2: South
      3: West

    The helicopter moves one cell in the chosen direction each step (with wrap-around).
    """

    def __init__(self, width=10, height=10, render_mode=None, max_steps=100, seed=0):
        automaton = ForestFire(width, height)
        super().__init__(automaton, render_mode)
        
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=5,  # Values 0-5 for the 6 possible cell states
                shape=(self.height, self.width),
                dtype=np.int32
            ),
            'position': spaces.Box(
                low=np.array([0, 0]), 
                high=np.array([self.height-1, self.width-1]),
                shape=(2,),
                dtype=np.int32
            )
        })
        self.action_space = spaces.Discrete(4)
        
        # Agent position
        self.agent_pos = np.array([self.height // 2, self.width // 2], dtype=int)
        # Step counter
        self.step_count = 0
        self.max_steps = max_steps

        self.state_colors = {
            self.automaton.EMPTY: (255, 255, 255),
            self.automaton.TREE: (0, 255, 0),
            self.automaton.FIRE_1: (255, 165, 0),
            self.automaton.FIRE_2: (255, 69, 0),
            self.automaton.FIRE_3: (255, 0, 0),
            self.automaton.ROCK: (105, 105, 105),
        }

        self.rng = np.random.RandomState(seed)

    def _reset_agent(self):
        # Place helicopter at center
        self.agent_pos = np.array([self.height // 2, self.width // 2], dtype=int)
        self.step_count = 0
        self._create_random_fire()

    def _create_random_fire(self):
        """Create a fire at a random tree location."""
        valid_positions = [
            (y, x)
            for y in range(self.height) for x in range(self.width)
            if (
                self.automaton.state[y, x] == self.automaton.TREE and
                # Distance from center (ie. the helicopter) must be at least 5 to prevent it from being too easy.
                abs(y - (self.height // 2)) + abs(x - (self.width // 2)) >= 5
            )
        ]
        assert len(valid_positions) > 0, "No tree positions found for setting fire."
        y, x = valid_positions[self.rng.randint(len(valid_positions))]
        self.automaton.state[y, x] = self.automaton.FIRE_1

    def step(self, action):
        extinguish_fire = self.automaton.state[self.agent_pos[0], self.agent_pos[1]] in (self.automaton.FIRE_1, self.automaton.FIRE_2, self.automaton.FIRE_3)
        if extinguish_fire:
            self.automaton.state[self.agent_pos[0], self.agent_pos[1]] = self.automaton.EMPTY
        self.automaton.step()
        if extinguish_fire:
            # Prevent e.g. tree from growing in the same frame.
            self.automaton.state[self.agent_pos[0], self.agent_pos[1]] = self.automaton.EMPTY

        self._move_agent(action)

        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._get_terminated()
        info = self._get_info()
        return obs, reward, terminated, False, info

    def _move_agent(self, action: int):
        directions = [
            (-1, 0),  # North
            (0, 1),  # East
            (1, 0),  # South
            (0, -1),  # West
        ]  # Northwest
        y, x = self.agent_pos
        dy, dx = directions[action]
        new_y = max(0, min(self.height - 1, y + dy))
        new_x = max(0, min(self.width - 1, x + dx))
        self.agent_pos = np.array([new_y, new_x], dtype=int)
        self.step_count += 1

    def _get_reward(self) -> float:
        state = self.automaton.get_state()
        num_trees = (state == self.automaton.TREE).sum()
        return float(num_trees) / 100.0

    def _get_observation(self) -> dict:
        """
        Return a dict: {'grid': np.ndarray of the automaton state, 'position': np.ndarray of shape (2,) with (y, x)}
        """
        grid = self.automaton.get_state().astype(np.int32)
        position = self.agent_pos.astype(np.int32)
        return {'grid': grid, 'position': position}

    # TODO: we should use truncation, not termination (maybe?)
    def _get_terminated(self) -> bool:
        # Episode ends after max_steps
        return self.step_count >= self.max_steps

    def _get_info(self) -> dict:
        return {"position": tuple(self.agent_pos)}

    def _render_agent(self, surface: pygame.Surface):
        # Draw helicopter as a blue triangle
        y, x = self.agent_pos
        cx = x * self.cell_size
        cy = y * self.cell_size
        size = self.cell_size

        # Draw an upward-pointing blue triangle centered in the cell
        point1 = (cx + size // 2, cy)  # top center
        point2 = (cx, cy + size)       # bottom left
        point3 = (cx + size, cy + size)  # bottom right
        pygame.draw.polygon(surface, (0, 0, 255), [point1, point2, point3])  # blue for helicopter