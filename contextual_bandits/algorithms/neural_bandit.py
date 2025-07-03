import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from contextual_bandits.algorithms.base import ContextualBandit


class NeuralNetwork(nn.Module):
    def __init__(self, context_dim: int, hidden_dims: list = [100, 50]):
        super(NeuralNetwork, self).__init__()

        layers = []
        input_dim = context_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralBandit(ContextualBandit):

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        hidden_dims: list = [100, 50],
        epsilon: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        random_state: t.Optional[int] = None,
    ):
        super().__init__(n_arms, context_dim, random_state)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if random_state is not None:
            torch.manual_seed(random_state)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [
            NeuralNetwork(context_dim, hidden_dims).to(self.device)
            for _ in range(n_arms)
        ]
        self.optimizers = [
            optim.Adam(model.parameters(), lr=learning_rate) for model in self.models
        ]
        self.loss_function = nn.MSELoss()

        self.X = [[] for _ in range(n_arms)]
        self.y = [[] for _ in range(n_arms)]

        # have models been trained?
        self.is_trained = [False] * n_arms

    def _context_to_tensor(self, context: np.ndarray) -> torch.Tensor:
        return torch.tensor(context, dtype=torch.float32, device=self.device)

    def select_arm(self, context: np.ndarray) -> int:
        context = np.array(context).reshape(-1)

        # epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)

        # exploitation
        rewards = np.zeros(self.n_arms)
        context_tensor = self._context_to_tensor(context)

        for arm in range(self.n_arms):
            if not self.is_trained[arm]:
                rewards[arm] = 0.5
            else:
                self.models[arm].eval()

                with torch.no_grad():
                    rewards[arm] = self.models[arm](context_tensor).item()

        return np.argmax(rewards)

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        context = np.array(context).reshape(-1)
        self.X[action].append(context)
        self.y[action].append(reward)

        if len(self.X[action]) >= self.batch_size:
            self._train_model(action)
            self.is_trained[action] = True

    def _train_model(self, arm: int) -> None:
        X_tensor = torch.tensor(np.array(self.X[arm]), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(self.y[arm]), dtype=torch.float32).view(-1, 1)

        self.models[arm].train()

        n_samples = len(np.array(self.X[arm]))
        n_batches = max(1, n_samples // self.batch_size)

        for _ in range(5):
            indices = np.random.permutation(n_samples)

            for i in range(n_batches):
                batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                predictions = self.models[arm](batch_X)
                loss = self.loss_function(predictions, batch_y)
                self.optimizers[arm].zero_grad()
                loss.backward()
                self.optimizers[arm].step()
