import abc
import numpy as np
import pygame
import os
from gymnasium import Env, spaces

from gym_cellular.cellular.automaton import CellularAutomaton


class AbstractCellularEnv(Env, abc.ABC):
    """
    Abstract base Gymnasium environment for a cellular automaton.
    Expects a CellularAutomaton instance to be passed in.
    Defines interface for reward(), terminated(), and update(action).
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 5}

    def __init__(self,
                 automaton: CellularAutomaton,
                 render_mode: str = None):
        super().__init__()
        self.automaton = automaton
        self.width = automaton.width
        self.height = automaton.height

        self.render_mode = render_mode
        self.window = None
        self.cell_size = 500 / max(self.width, self.height)

    @abc.abstractmethod
    def _get_observation(self) -> dict:
        """
        Get the observation of the current state.
        """
        pass

    @abc.abstractmethod
    def step(self, action):
        """
        Take a step in the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        # Optionally, reseed automaton or environment. For simplicity, reset automaton state randomly.
        if seed is not None:
            np.random.seed(seed)
        self.automaton.reset()
        # Reset agent position in concrete
        self._reset_agent()
        obs = self._get_observation()
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
        grid = self.automaton.get_state()
        for y in range(self.height):
            for x in range(self.width):
                cell_val = int(grid[y, x])
                color = self.state_colors.get(cell_val, (0, 0, 0))
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(surface, color, rect)
        # Draw agent
        self._render_agent(surface)
        # Blit to window
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.time.Clock().tick(self.metadata.get("render_fps", 5))

    def save_frame(self, filename="frame.png"):
        """
        Save the current frame to a file.
        
        Args:
            filename (str): The name of the file to save to. Default is 'frame.png'.
        """
        if self.window is None:
            # Render a frame if window doesn't exist
            self.render()
            if self.render_mode != "human" or self.window is None:
                print("Cannot save frame: rendering is disabled or failed")
                return False
                
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Get the surface from the window and save it
        surface = pygame.Surface(self.window.get_size())
        surface.blit(self.window, (0, 0))
        pygame.image.save(surface, filename)
        return True

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None