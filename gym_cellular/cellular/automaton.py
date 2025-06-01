import abc
import numpy as np


class CellularAutomaton(abc.ABC):
    """
    Abstract base class for cellular automata.
    Defines interface for initializing, stepping, and accessing state.
    """

    @abc.abstractmethod
    def __init__(self, width: int, height: int):
        """
        Initialize the automaton with given grid dimensions.
        Should set up self.width, self.height, and self.state (2D numpy array).
        """
        self.width = width
        self.height = height
        self.state = np.zeros((height, width), dtype=np.uint8)

    @abc.abstractmethod
    def step(self):
        """
        Advance the automaton by one time step, updating self.state.
        """
        pass

    @abc.abstractmethod
    def possible_states(self) -> list[int]:
        """
        Return a list of all integer values that cells can take in this CA.
        For example, GameOfLife returns [0,1]; ForestFire returns [0,1,2,3,4,5].
        """
        pass

    def get_state(self) -> np.ndarray:
        """
        Returns the current state of the grid as a 2D numpy array.
        """
        return self.state.copy()

    def set_state(self, new_state: np.ndarray):
        """
        Replace the current state with new_state. Should match dimensions.
        """
        assert new_state.shape == (self.height, self.width)
        self.state = new_state.copy()

    @abc.abstractmethod
    def reset(self):
        """
        Reset the automaton to its initial state.
        """
        pass
