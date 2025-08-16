from abc import ABC, abstractmethod
import numpy as np

class BaseRLEnvironment(ABC):
    
    def __init__(self):
        self.n_states: int = None
        self.n_actions: int = None
        self.policy: np.ndarray = None
        self.V: np.ndarray = None
        self.Q: np.ndarray = None
        self.N: np.ndarray = None

    @abstractmethod
    def sample_trajectory(self, epsilon_greedy: bool = True, epsilon: float = 0.1, max_steps: int = 10000) -> list[tuple[int, int, float]]:
        pass