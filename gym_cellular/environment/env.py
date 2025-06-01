import abc
import numpy as np
import pygame
from gymnasium import Env, spaces

from gym_cellular.cellular.automaton import CellularAutomaton


class AbstractCellularEnv(Env, abc.ABC):
    """
    Abstract base Gymnasium environment for a cellular automaton.
    Expects a CellularAutomaton instance to be passed in.
    Defines interface for reward(), terminated(), and update(action).
    """
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self,
                 automaton: CellularAutomaton,
                 render_mode: str = None):
        super().__init__()
        self.automaton = automaton
        self.width = automaton.width
        self.height = automaton.height

        # Placeholder spaces; concrete envs should override
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.height, self.width),
            dtype=np.uint8
        )

        self.render_mode = render_mode
        self.window = None
        self.cell_size = 10  # pixels

    @abc.abstractmethod
    def _get_reward(self) -> float:
        """
        Compute and return the reward for the current state.
        """
        pass

    @abc.abstractmethod
    def _get_terminated(self) -> bool:
        """
        Determine whether the episode is terminated.
        """
        pass

    @abc.abstractmethod
    def _get_info(self) -> dict:
        """
        Return any additional info.
        """
        pass

    @abc.abstractmethod
    def _move_agent(self, action: int):
        """
        Move the agent (e.g., helicopter) based on the action.
        """
        pass

    def step(self, action):
        # Move agent according to action
        self._move_agent(action)
        # Advance automaton
        self.automaton.step()
        obs = self.automaton.get_state()
        reward = self._get_reward()
        terminated = self._get_terminated()
        info = self._get_info()
        return obs, reward, terminated, False, info
        # Note: Gymnasium expects (obs, reward, terminated, truncated, info)

    def reset(self, seed=None, options=None):
        # Optionally, reseed automaton or environment. For simplicity, reset automaton state randomly.
        if seed is not None:
            np.random.seed(seed)
        # If automaton supports random init, reinitialize
        if hasattr(self.automaton, 'state'):
            # Randomize state
            self.automaton.state = np.random.randint(2, size=(self.height, self.width), dtype=np.uint8)
        # Reset agent position in concrete
        self._reset_agent()
        obs = self.automaton.get_state()
        return obs, {}

    @abc.abstractmethod
    def _reset_agent(self):
        """Reset the agent (e.g., helicopter) to initial position."""
        pass

    def render(self):
        if self.render_mode != "human":
            return
        # Lazy setup
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            pygame.display.set_caption("Cellular Automaton Env")
        # Draw grid
        surface = pygame.Surface(self.window.get_size())
        surface.fill((0, 0, 0))  # black background
        state = self.automaton.get_state()
        for y in range(self.height):
            for x in range(self.width):
                if state[y, x] == 1:
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(surface, (255, 255, 255), rect)  # white for alive cells
        # Draw agent
        self._render_agent(surface)
        # Blit to window
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.time.Clock().tick(self.metadata.get("render_fps", 5))

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None