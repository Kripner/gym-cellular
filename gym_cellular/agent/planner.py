import numpy as np
from abc import ABC, abstractmethod

from gym_cellular.cellular.forest_fire import ForestFire


class WorldModel(ABC):
    """
    Abstract interface for a world model that, given the current grid
    state (2D np.ndarray), predicts the next grid state (2D np.ndarray).
    """

    @abstractmethod
    def predict(self, state: dict) -> np.ndarray:
        """
        Given a 2D array `state` of shape (H, W), return a new 2D array of
        the same shape representing the next time step under this model.
        """
        pass

class OracleWorldModel(WorldModel):
    """
    “Perfect” world‐model: wraps an actual CellularAutomaton instance.
    On each predict(state) call, it overwrites the automaton’s internal
    state and runs `step()`, returning the true next state.
    """

    def __init__(self, automaton):
        """
        `automaton` must be a freshly created CellularAutomaton (e.g. GameOfLife or ForestFire),
        whose width/height match the environment. We will never call its constructor again—
        but each predict() call will do `set_state(state)` then `step()`.
        """
        # We assume `automaton` has methods: set_state(np.ndarray), step(), and get_state().
        self.automaton = automaton

    def predict(self, state: dict) -> np.ndarray:
        new_grid = state["grid"].copy()
        y, x = state["position"]
        cell_val = state["grid"][y, x]
        if cell_val in (ForestFire.FIRE_1, ForestFire.FIRE_2, ForestFire.FIRE_3):
            new_grid[y, x] = ForestFire.EMPTY

        self.automaton.set_state(new_grid)
        self.automaton.step()
        return self.automaton.get_state()

class StaticWorldModel(WorldModel):
    """
    “Naive” world‐model: assumes the grid never changes.
    predict(state) → return a copy of `state` itself.
    """

    def predict(self, state: dict) -> np.ndarray:
        return state["grid"].copy()


class PlanningAgent:
    """
    A depth‐d tree‐search agent for the HelicopterEnv.  At each decision step,
    it knows the current grid (2D np.ndarray of integers), the current agent_pos
    (y, x), and a `WorldModel` (which can predict grid → next grid).  It chooses
    the action (0–7) whose resulting “look‐ahead” (up to depth d) yields the
    maximum total count of cells == 1 (“trees”) at the leaf nodes.

    Usage:
        agent = PlanningAgent(depth=d, world_model=some_model, height=H, width=W)
        action = agent.select_action(current_grid, (agent_y, agent_x))
    """

    def __init__(self, depth: int, world_model: WorldModel, height: int, width: int, seed: int = 0):
        self.depth = depth
        self.model = world_model
        self.height = height
        self.width = width
        self.rng = np.random.RandomState(seed)

        self.directions = [
            (-1, 0),  # N
            (0, 1),  # E
            (1, 0),  # S
            (0, -1),  # W
        ]

    def select_action(self, state: np.ndarray, agent_pos: tuple[int, int]) -> int:
        """
        Given the current `state` (2D array) and `agent_pos = (y, x)`,
        return the best action ∈ {0,...,3} by doing a depth-d look‐ahead
        and maximizing the cumulative reward.
        """
        best_action, _ = self._search(state, agent_pos, self.depth, 0)
        return best_action

    def _search(
            self,
            grid: np.ndarray,
            pos: tuple[int, int],
            depth_remaining: int,
            total_reward: int,
    ) -> tuple[int | None, int]:
        assert depth_remaining >= 0
        if depth_remaining == 0:
            fire_positions = [
                (y, x) for y in range(grid.shape[0]) for x in range(grid.shape[1]) 
                if grid[y, x] in (ForestFire.FIRE_1, ForestFire.FIRE_2, ForestFire.FIRE_3)
            ]
            if fire_positions:
                min_fire_distance = min(abs(pos[0] - fy) + abs(pos[1] - fx) for fy, fx in fire_positions)
                proximity_bonus = max(0, 10 - min_fire_distance)
                total_reward += proximity_bonus
            return None, total_reward

        next_grid = self.model.predict({
            "grid": grid,
            "position": pos,
        })
        total_reward += int(np.sum(next_grid == ForestFire.TREE))

        best_action, best_score = None, None
        for action in self.rng.permutation(4):
            y, x = pos
            dy, dx = self.directions[action]
            new_y = max(0, min(self.height - 1, y + dy))
            new_x = max(0, min(self.width - 1, x + dx))

            _, score = self._search(
                next_grid,
                (new_y, new_x),
                depth_remaining - 1,
                total_reward,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score
