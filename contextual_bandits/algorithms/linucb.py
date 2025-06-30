import numpy as np
from contextual_bandits.algorithms.base import ContextualBandit


class LinUCB(ContextualBandit):
    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        random_state: int = None,
    ):
        super().__init__(n_arms, context_dim, random_state)
        self.alpha = alpha
        self.A = [np.identity(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        context = np.array(context).reshape(-1)
        ucb_values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[arm], np.eye(self.context_dim))
            self.theta[arm] = A_inv @ self.b[arm]
            expected_reward = context.dot(self.theta[arm])
            cb = self.alpha * np.sqrt(context.dot(A_inv).dot(context))
            ucb_values[arm] = expected_reward + cb

        best_arm = np.argmax(ucb_values)
        return best_arm

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        context = np.array(context).reshape(-1)
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
