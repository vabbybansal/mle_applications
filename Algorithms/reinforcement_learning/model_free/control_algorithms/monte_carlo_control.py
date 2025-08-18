import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problems.BaseRLEnvironment import BaseRLEnvironment
from problems.model_free_frozen_lake import ModelFreeFrozenLake

class MonteCarloControl:
    def __init__(
        self,
        env: BaseRLEnvironment,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        gamma: float = 0.99,
        alpha: float = 0.1,
        alpha_decay: float = 0.995,
        alpha_min: float = 0.01,
    ):
        self.env = env
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        # self.env.N = np.zeros((self.env.n_states, self.env.n_actions))

        self.logger = {
            "trajectories": [],
            "trajectory_lengths": [],
            "trajectory_returns": [],
            "success_rates": [],  # Ensure this is always present
            "gamma_returns": [],  # New: stores average gamma returns at checkpoints
            "exploration_flags": [],  # New: stores whether each trajectory was exploratory
            "epsilon_values": [],  # New: stores epsilon at each iteration
            "alpha_values": [],    # New: stores alpha at each iteration (for plotting)
            "best_policy_iteration": None,  # New: store best policy iteration for plotting
            "eval_avg_traj_lengths": [],  # New: avg trajectory length from evaluation at checkpoints
        }

    def epsilon_greedy_policy_update(self, Q, epsilon):
        nA = Q.shape[1]
        policy = np.ones_like(Q) * (epsilon / nA)
        best_actions = np.argmax(Q, axis=1)
        for s in range(Q.shape[0]):
            policy[s][best_actions[s]] += 1 - epsilon
        return policy

    def epsilon_greedy_improvement(self):
        # decay epsilon (GLIE)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        # decay alpha
        self.alpha *= self.alpha_decay
        self.alpha = max(self.alpha, self.alpha_min)
        # update the policy using epsilon greedy (Note: this is not needed since for sampling, we directly use Q table)
        # self.env.policy = self.epsilon_greedy_policy_update(self.env.Q, self.epsilon)

    def evaluation(self, trajectory: list[tuple[int, int, float]]):
        # Reward to go from time step t to the end of the trajectory
        G = 0.0
        # for each state and action, update the value function (Every Visit Monte Carlo)
        for state, action, reward in reversed(trajectory):
            G = self.gamma * G + reward
            # self.env.N[state][action] += 1  # we use alpha instead of 1/N
            # converting 1/N to alpha was super important. WHY? TODO: research.
            self.env.Q[state][action] = self.env.Q[state][action] + self.alpha * (G - self.env.Q[state][action])
        # Log the return of the trajectory (from the start state)
        self.logger["trajectory_returns"].append(round(G, 2))

    def fit(self, num_iterations=1000) -> dict:
        best_success = -1
        best_gamma_return = float('-inf')
        best_Q = None
        best_policy_iteration = None
        best_avg_gamma_return = None
        metrics = {}

        for i in range(num_iterations):
            # sample a single trajectory
            trajectory, is_sample_exploration = self.env.sample_trajectory(epsilon_greedy=True, epsilon=self.epsilon, max_steps=200)

            self.logger["trajectories"].append(trajectory)
            self.logger["exploration_flags"].append(is_sample_exploration)  # Log exploration
            self.logger["epsilon_values"].append(self.epsilon)  # Log epsilon
            self.logger["alpha_values"].append(self.alpha)      # Log alpha (even if decaying)

            # evaluate trajectory (also logs return)
            self.evaluation(trajectory)

            # improve policy and decay epsilon/alpha
            self.epsilon_greedy_improvement()

            if i % 100 != 0:
                # log -1.0 for all intermediate iterations
                self.logger["success_rates"].append(-1.0)
                self.logger["gamma_returns"].append(-1.0)
                self.logger["eval_avg_traj_lengths"].append(-1.0)

            # policy checkpoint
            if i % 100 == 0:  # evaluate every 100 episodes
                metrics_dict = self.env.evaluate_policy_metrics(num_episodes=50, gamma=self.gamma)
                success = metrics_dict["success_rate"]
                avg_gamma_return = metrics_dict["avg_gamma_return"]
                avg_traj_len_eval = metrics_dict.get("avg_trajectory_length", None)
                self.logger["success_rates"].append(success)
                self.logger["gamma_returns"].append(avg_gamma_return)
                self.logger["eval_avg_traj_lengths"].append(avg_traj_len_eval if avg_traj_len_eval is not None else -1.0)
                if avg_gamma_return > best_gamma_return:
                    best_gamma_return = avg_gamma_return
                    best_Q = self.env.Q.copy()
                    best_policy_iteration = i
                    best_avg_gamma_return = avg_gamma_return
        # Store the best policy iteration for plotting
        self.logger["best_policy_iteration"] = best_policy_iteration
        
        if best_Q is not None:
            self.env.Q = best_Q

        final_metrics = self.env.evaluate_policy_metrics(num_episodes=50, gamma=self.gamma)
        final_policy_success_rate = final_metrics["success_rate"]
        final_policy_avg_gamma_return = final_metrics["avg_gamma_return"]
        final_policy_avg_traj_length = final_metrics.get("avg_trajectory_length", 0.0)

        metrics["final_policy_success_rate"] = final_policy_success_rate
        metrics["final_policy_avg_gamma_return"] = final_policy_avg_gamma_return
        metrics["best_policy_iteration"] = best_policy_iteration
        metrics["avg_trajectory_length"] = final_policy_avg_traj_length

        # Store summary/final metrics in logger for plotting
        self.logger["final_metrics"] = {
            "final_policy_success_rate": final_policy_success_rate,
            "final_policy_avg_gamma_return": final_policy_avg_gamma_return,
            "avg_trajectory_length": final_policy_avg_traj_length,
            "best_policy_iteration": best_policy_iteration,
            "best_policy_success_rate": best_success,
            "best_policy_avg_gamma_return": best_avg_gamma_return,
        }

        return metrics

def run_monte_carlo_control_varying_gamma(
    gammas = None,
    num_runs_per_gamma = 10,
    num_iterations = 1000,
    epsilon = 0.3,
    epsilon_decay = 0.995,
    alpha=0.1,
    alpha_decay=0.995,
    alpha_min=0.01
):
    """
    Runs Monte Carlo Control for different gamma values, each averaged over several runs,
    and plots the average trajectory return curve for each gamma.
    """

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))

    for idx, gamma in enumerate(gammas):
        print(f"gamma: {gamma}, id: {idx} / {len(gammas)}")
        all_returns = []
        for run in range(num_runs_per_gamma):
            env = ModelFreeFrozenLake()
            mc_control = MonteCarloControl(
                env,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                gamma=gamma,
                alpha=alpha,
                alpha_decay=alpha_decay,
                alpha_min=alpha_min
            )
            mc_control.fit(num_iterations=num_iterations)
            all_returns.append(mc_control.logger["trajectory_returns"])
        avg_returns = np.mean(all_returns, axis=0)
        plt.plot(
            range(len(avg_returns)),
            avg_returns,
            label=f"gamma={gamma:.2f} (return)",
            color=colors[idx],
            linestyle='-',
            alpha=0.7
        )

    plt.xlabel("Iteration")
    plt.ylabel("Average Trajectory Return")
    plt.title(f"Monte Carlo Control: Trajectory Returns for Different Gamma Values\n(Averaged over {num_runs_per_gamma} runs)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = ModelFreeFrozenLake(
        step_penalty=-0.001,
        hole_penalty=-5.0,
        goal_reward=20.0,
    )
    monte_carlo_control = MonteCarloControl(
        env,
        epsilon=0.9,
        epsilon_decay=0.999,
        epsilon_min=0.001,
        gamma=0.99,
        alpha=0.5,
        alpha_decay=0.999,
        alpha_min=0.001
    )
    metrics = monte_carlo_control.fit(num_iterations=10000)
    print("**************************************************")
    print("Metrics: ", metrics)
    print("**************************************************")
    # print(monte_carlo_control.env.Q)
    env.plot_training_stats(monte_carlo_control.logger, smooth_window=50)
    env.plot_q_heatmap(title="Q-Value Heatmap (final Q)", show=False)
    plt.title("Monte Carlo Control")
    plt.show()
