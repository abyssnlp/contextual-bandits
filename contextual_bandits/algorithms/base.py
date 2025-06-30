from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class ContextualBandit(ABC):
    def __init__(
        self, n_arms: int, context_dim: int, random_state: Optional[int] = None
    ):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.actions_history = []
        self.contexts_history = []
        self.rewards_history = []

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        pass

    def fit(
        self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> None:
        for i in range(len(rewards)):
            self.update(contexts[i], actions[i], rewards[i])

    def add_to_history(self, context: np.ndarray, action: int, reward: float) -> None:
        self.contexts_history.append(context)
        self.actions_history.append(action)
        self.rewards_history.append(reward)

    def get_history(self):
        return {
            "contexts": self.contexts_history,
            "actions": self.actions_history,
            "rewards": self.rewards_history,
        }
