"""
AI generated script to run the algorithms for simulation for marketing campaign selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from generate_data import generate_marketing_data
from contextual_bandits.algorithms.linucb import LinUCB
from contextual_bandits.algorithms.neural_bandit import NeuralBandit
from contextual_bandits.utils.preprocessing import create_preprocesser


class MarketingEnv:
    """
    Environment wrapper for marketing campaign simulation.
    Provides proper reward generation for any action at any context.
    """

    def __init__(
        self, n_campaigns: int = 5, n_user_features: int = 10, random_state: int = 42
    ):
        self.n_campaigns = n_campaigns
        self.n_user_features = n_user_features
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Generate campaign parameters that define conversion probabilities
        self.campaign_params = self.rng.randn(n_campaigns, n_user_features)

        # Current context
        self.current_context = None

    def sample_context(self) -> np.ndarray:
        """Sample a new user context (user features)."""
        context = self.rng.randn(self.n_user_features)
        self.current_context = context
        return context

    def get_conversion_probability(self, context: np.ndarray, action: int) -> float:
        """Get the true conversion probability for a given context and action."""
        logit = np.dot(context, self.campaign_params[action])
        return 1.0 / (1.0 + np.exp(-logit))

    def get_reward(self, context: np.ndarray, action: int) -> float:
        """Sample a reward for the given context and action."""
        prob = self.get_conversion_probability(context, action)
        return float(self.rng.binomial(1, prob))

    def get_optimal_action(self, context: np.ndarray) -> int:
        """Get the optimal action for a given context."""
        probs = [
            self.get_conversion_probability(context, a) for a in range(self.n_campaigns)
        ]
        return int(np.argmax(probs))

    def get_optimal_reward(self, context: np.ndarray) -> float:
        """Get the optimal expected reward for a given context."""
        optimal_action = self.get_optimal_action(context)
        return self.get_conversion_probability(context, optimal_action)


class BanditSimulation:
    """
    A simulation class for contextual bandit algorithms in marketing scenarios.
    """

    def __init__(
        self, n_campaigns: int = 5, n_user_features: int = 10, random_state: int = 42
    ):
        self.n_campaigns = n_campaigns
        self.n_user_features = n_user_features
        self.random_state = random_state
        self.results = {}
        self.env = MarketingEnv(n_campaigns, n_user_features, random_state)
        self.preprocessor = None

    def prepare_data(
        self, n_samples: int = 2000, train_ratio: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate and preprocess marketing data for simulation.
        Fixed to avoid data leakage by fitting preprocessor only on training data.

        Returns:
            train_contexts: Training contexts (preprocessed)
            test_contexts: Test contexts (preprocessed)
            train_actions: Training actions (for reference only, not used in true online simulation)
            test_actions: Test actions (for reference only)
            test_rewards: Test rewards (for reference only)
        """
        logger.info(f"Generating {n_samples} marketing samples...")

        # we'll use our environment for true reward generation
        df = generate_marketing_data(
            n_samples=n_samples,
            n_campaigns=self.n_campaigns,
            n_user_features=self.n_user_features,
            random_state=self.random_state,
        )

        # Define feature columns
        numerical_features = [
            f"user_feature_{i}" for i in range(self.n_user_features)
        ] + [
            "previous_purchases",
            "day_of_week",
            "hour_of_day",
            "email_opened",
            "email_clicked",
        ]

        categorical_features = ["age_group", "gender", "device"]

        # Prepare features for preprocessing
        feature_columns = numerical_features + categorical_features
        X = df[feature_columns]

        # Split data first to avoid data leakage
        train_size = int(n_samples * train_ratio)
        X_train = X[:train_size]
        X_test = X[train_size:]

        # Create and fit preprocessor ONLY on training data
        self.preprocessor = create_preprocesser(
            categorical_features, numerical_features
        )
        train_contexts = self.preprocessor.fit_transform(X_train)

        # Transform test data using the fitted preprocessor
        test_contexts = self.preprocessor.transform(X_test)

        # Extract actions and rewards (for reference, but we'll use environment for true simulation)
        campaign_to_idx = {f"campaign_{i}": i for i in range(self.n_campaigns)}
        train_actions = df["campaign_id"][:train_size].map(campaign_to_idx).values
        test_actions = df["campaign_id"][train_size:].map(campaign_to_idx).values
        test_rewards = df["conversion"][train_size:].values.astype(float)

        logger.info(
            f"Data prepared: {train_contexts.shape[0]} train samples, {test_contexts.shape[0]} test samples"
        )
        logger.info(f"Feature dimension: {train_contexts.shape[1]}")
        logger.info(
            f"Training campaign distribution: {np.bincount(train_actions, minlength=self.n_campaigns)}"
        )
        logger.info(
            f"Training conversion rate: {df['conversion'][:train_size].mean():.3f}"
        )

        return train_contexts, test_contexts, train_actions, test_actions, test_rewards

    def run_simulation(
        self,
        train_contexts: np.ndarray,
        test_contexts: np.ndarray,
        train_actions: np.ndarray,
        test_actions: np.ndarray,
        test_rewards: np.ndarray,
        use_pretrain: bool = False,
    ) -> Dict:
        """
        Run the contextual bandit simulation with proper online evaluation.

        Args:
            train_contexts: Training contexts (for optional pretraining)
            test_contexts: Test contexts for online simulation
            train_actions: Training actions (for optional pretraining)
            test_actions: Test actions (for reference only)
            test_rewards: Test rewards (for reference only)
            use_pretrain: Whether to pretrain algorithms (not recommended due to bias)

        Returns:
            Dictionary containing results for each algorithm
        """
        n_test_samples = len(test_contexts)
        context_dim = test_contexts.shape[1]

        logger.info(f"Running simulation with {n_test_samples} online samples...")
        if use_pretrain:
            logger.warning(
                "Using pretraining - this may introduce bias from logged policy"
            )

        # Initialize algorithms
        algorithms = {
            "LinUCB": LinUCB(
                n_arms=self.n_campaigns,
                context_dim=context_dim,
                alpha=1.0,
                random_state=self.random_state,
            ),
            "Neural Bandit": NeuralBandit(
                n_arms=self.n_campaigns,
                context_dim=context_dim,
                epsilon=0.1,
                learning_rate=0.001,
                batch_size=32,
                random_state=self.random_state,
            ),
        }

        # Track results
        results = {
            name: {
                "rewards": [],
                "cumulative_rewards": [],
                "regret": [],
                "actions_taken": [],
                "optimal_actions": [],
                "optimal_rewards": [],
            }
            for name in algorithms.keys()
        }

        # Add random baseline
        results["Random"] = {
            "rewards": [],
            "cumulative_rewards": [],
            "regret": [],
            "actions_taken": [],
            "optimal_actions": [],
            "optimal_rewards": [],
        }

        # Optional pre-training (not recommended due to bias)
        if use_pretrain:
            logger.info("Pre-training algorithms on logged data...")
            for name, algorithm in algorithms.items():
                # Create fake rewards for training using environment
                train_rewards = []
                for i in range(len(train_contexts)):
                    reward = self.env.get_reward(train_contexts[i], train_actions[i])
                    train_rewards.append(reward)

                algorithm.fit(train_contexts, train_actions, np.array(train_rewards))

        # Convert raw user features from test contexts back for environment
        # Since test_contexts are preprocessed, we need to map them back to raw features
        # For simplicity, we'll generate new contexts from the environment
        logger.info("Running online simulation...")

        rng = np.random.RandomState(
            self.random_state + 1
        )  # Different seed for simulation

        for t in range(n_test_samples):
            # Generate new context from environment
            raw_context = self.env.sample_context()

            # Preprocess context using fitted preprocessor
            # Create a temporary dataframe with the same structure as training data
            context_dict = {
                f"user_feature_{i}": raw_context[i] for i in range(self.n_user_features)
            }

            context_dict.update(
                {
                    "age_group": rng.choice(
                        ["18-24", "25-34", "35-44", "45-54", "55+"]
                    ),
                    "gender": rng.choice(["M", "F", "Other"]),
                    "device": rng.choice(["Mobile", "Desktop", "Tablet"]),
                    "previous_purchases": rng.poisson(2),
                    "day_of_week": rng.randint(0, 7),
                    "hour_of_day": rng.randint(0, 24),
                    "email_opened": rng.binomial(1, 0.3),
                    "email_clicked": rng.binomial(1, 0.2),
                }
            )

            context_df = pd.DataFrame([context_dict])
            preprocessed_context = self.preprocessor.transform(context_df)[0]

            # Calculate optimal action and reward for this context
            optimal_action = self.env.get_optimal_action(raw_context)
            optimal_reward = self.env.get_optimal_reward(raw_context)

            if t % 500 == 0:
                logger.info(f"Step {t}/{n_test_samples}")

            # For each algorithm
            for name, algorithm in algorithms.items():
                # Select action based on preprocessed context
                selected_action = algorithm.select_arm(preprocessed_context)

                # Get true reward from environment
                observed_reward = self.env.get_reward(raw_context, selected_action)

                # Update algorithm with observed reward
                algorithm.update(preprocessed_context, selected_action, observed_reward)

                # Track metrics
                results[name]["rewards"].append(observed_reward)
                results[name]["actions_taken"].append(selected_action)
                results[name]["optimal_actions"].append(optimal_action)
                results[name]["optimal_rewards"].append(optimal_reward)

                # Calculate cumulative rewards
                if len(results[name]["cumulative_rewards"]) == 0:
                    results[name]["cumulative_rewards"].append(observed_reward)
                else:
                    results[name]["cumulative_rewards"].append(
                        results[name]["cumulative_rewards"][-1] + observed_reward
                    )

                # Calculate regret (optimal_reward - observed_reward)
                regret = optimal_reward - observed_reward
                results[name]["regret"].append(regret)

            # Random baseline
            random_action = rng.randint(self.n_campaigns)
            random_reward = self.env.get_reward(raw_context, random_action)

            results["Random"]["rewards"].append(random_reward)
            results["Random"]["actions_taken"].append(random_action)
            results["Random"]["optimal_actions"].append(optimal_action)
            results["Random"]["optimal_rewards"].append(optimal_reward)

            if len(results["Random"]["cumulative_rewards"]) == 0:
                results["Random"]["cumulative_rewards"].append(random_reward)
            else:
                results["Random"]["cumulative_rewards"].append(
                    results["Random"]["cumulative_rewards"][-1] + random_reward
                )

            regret = optimal_reward - random_reward
            results["Random"]["regret"].append(regret)

        self.results = results
        return results

    def plot_results(self) -> None:
        """
        Create comprehensive plots of the simulation results.
        """
        if not self.results:
            print("No results to plot. Run simulation first.")
            return

        # Use a more reliable matplotlib style
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            plt.style.use("default")
            logger.warning("seaborn style not available, using default")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Contextual Bandit Simulation Results: Marketing Campaign Selection",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Cumulative Rewards
        ax1 = axes[0, 0]
        for name, data in self.results.items():
            ax1.plot(data["cumulative_rewards"], label=name, linewidth=2)
        ax1.set_title("Cumulative Rewards Over Time", fontweight="bold")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Cumulative Rewards")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative Regret
        ax2 = axes[0, 1]
        for name, data in self.results.items():
            cumulative_regret = np.cumsum(data["regret"])
            ax2.plot(cumulative_regret, label=name, linewidth=2)
        ax2.set_title("Cumulative Regret Over Time", fontweight="bold")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Cumulative Regret")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Average Reward (Moving Window)
        ax3 = axes[0, 2]
        window_size = 50
        for name, data in self.results.items():
            if len(data["rewards"]) >= window_size:
                moving_avg = (
                    pd.Series(data["rewards"]).rolling(window=window_size).mean()
                )
                ax3.plot(moving_avg, label=name, linewidth=2)
        ax3.set_title(
            f"Moving Average Reward (Window={window_size})", fontweight="bold"
        )
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Average Reward")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Action Selection Distribution
        ax4 = axes[1, 0]
        action_data = []
        algorithms = []
        for name, data in self.results.items():
            action_counts = np.bincount(
                data["actions_taken"], minlength=self.n_campaigns
            )
            action_data.append(action_counts)
            algorithms.append(name)

        action_df = pd.DataFrame(
            action_data,
            columns=[f"Campaign {i}" for i in range(self.n_campaigns)],
            index=algorithms,
        )
        action_df.plot(kind="bar", ax=ax4, stacked=True)
        ax4.set_title("Action Selection Distribution", fontweight="bold")
        ax4.set_xlabel("Algorithm")
        ax4.set_ylabel("Number of Selections")
        ax4.legend(title="Campaign", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.tick_params(axis="x", rotation=45)

        # 5. Optimal Action Accuracy (instead of just "Accuracy")
        ax5 = axes[1, 1]
        accuracies = {}
        for name, data in self.results.items():
            correct_actions = np.array(data["actions_taken"]) == np.array(
                data["optimal_actions"]
            )
            accuracy = correct_actions.mean()
            accuracies[name] = accuracy

        algorithms = list(accuracies.keys())
        accuracy_values = list(accuracies.values())
        bars = ax5.bar(
            algorithms,
            accuracy_values,
            color=["skyblue", "lightcoral", "lightgreen"][: len(algorithms)],
        )
        ax5.set_title("Optimal Action Selection Accuracy", fontweight="bold")
        ax5.set_xlabel("Algorithm")
        ax5.set_ylabel("Accuracy")
        ax5.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, accuracy_values):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax5.tick_params(axis="x", rotation=45)
        ax5.grid(True, alpha=0.3, axis="y")

        # 6. Final Performance Summary
        ax6 = axes[1, 2]
        metrics = ["Total Rewards", "Avg Reward", "Final Regret"]
        performance_data = []

        for name, data in self.results.items():
            total_rewards = sum(data["rewards"])
            avg_reward = np.mean(data["rewards"])
            final_regret = sum(data["regret"])
            performance_data.append([total_rewards, avg_reward, final_regret])

        performance_df = pd.DataFrame(
            performance_data, columns=metrics, index=list(self.results.keys())
        )

        # Normalize for comparison
        performance_df_norm = performance_df.div(performance_df.max())
        performance_df_norm["Final Regret"] = (
            1 - performance_df_norm["Final Regret"]
        )  # Lower is better

        x = np.arange(len(metrics))
        width = 0.25

        for i, (name, values) in enumerate(performance_df_norm.iterrows()):
            ax6.bar(x + i * width, values, width, label=name, alpha=0.8)

        ax6.set_title("Normalized Performance Comparison", fontweight="bold")
        ax6.set_xlabel("Metrics")
        ax6.set_ylabel("Normalized Score")
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(metrics)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        self.print_summary()

    def print_summary(self) -> None:
        """
        Print a summary of the simulation results.
        """
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)

        for name, data in self.results.items():
            total_rewards = sum(data["rewards"])
            avg_reward = np.mean(data["rewards"])
            total_regret = sum(data["regret"])
            avg_regret = np.mean(data["regret"])

            # Calculate accuracy using optimal actions
            correct_actions = np.array(data["actions_taken"]) == np.array(
                data["optimal_actions"]
            )
            accuracy = correct_actions.mean()

            print(f"\n{name}:")
            print(f"  Total Rewards: {total_rewards:.2f}")
            print(f"  Average Reward: {avg_reward:.4f}")
            print(f"  Total Regret: {total_regret:.2f}")
            print(f"  Average Regret: {avg_regret:.4f}")
            print(f"  Optimal Action Accuracy: {accuracy:.4f}")

            # Action distribution
            action_counts = np.bincount(
                data["actions_taken"], minlength=self.n_campaigns
            )
            action_probs = action_counts / sum(action_counts)
            print(f"  Action Distribution: {[f'{p:.3f}' for p in action_probs]}")

            # Expected optimal reward
            avg_optimal_reward = np.mean(data["optimal_rewards"])
            print(f"  Average Optimal Reward: {avg_optimal_reward:.4f}")
            print(
                f"  Regret Rate: {avg_regret/avg_optimal_reward:.4f}"
                if avg_optimal_reward > 0
                else "  Regret Rate: N/A"
            )


def main():
    """
    Main function to run the contextual bandit simulation.
    """
    logger.info("Starting Contextual Bandit Simulation for Marketing Campaigns")
    logger.info("=" * 60)

    # Initialize simulation
    simulation = BanditSimulation(n_campaigns=5, n_user_features=10, random_state=42)

    # Generate and prepare data (with proper train/test split to avoid leakage)
    train_contexts, test_contexts, train_actions, test_actions, test_rewards = (
        simulation.prepare_data(n_samples=5000, train_ratio=0.3)
    )

    # Run simulation (use_pretrain=False to avoid bias from logged policy)
    results = simulation.run_simulation(
        train_contexts,
        test_contexts,
        train_actions,
        test_actions,
        test_rewards,
        use_pretrain=False,
    )

    # Plot results
    simulation.plot_results()

    logger.info("Simulation completed successfully!")


if __name__ == "__main__":
    main()
