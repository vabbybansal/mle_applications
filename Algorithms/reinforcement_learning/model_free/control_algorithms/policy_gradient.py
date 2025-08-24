import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problems.BaseRLEnvironment import BaseRLEnvironment
from problems.model_free_frozen_lake import ModelFreeFrozenLake
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import ray

class Pi(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=128):
        super(Pi, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        logits = self.fc2(self.act(self.fc1(x)))
        return logits
    
class PolicyGradient:
    def __init__(
        self,
        env: BaseRLEnvironment,
        gamma: float = 0.99,
        step_size: float = 0.001,
        hidden: int = 20,
    ):
        self.env = env
        self.pi = Pi(env.n_states, env.n_actions, hidden)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=step_size)
        self.gamma = gamma
        self.logger = {"success_rates": [], "gamma_returns": [], "eval_avg_traj_lengths": []}
        
        # Initialize plot for real-time updates
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def update_policy(self, trajectory_batch: list[list[tuple[int, int, float]]]):
        
        loss = torch.tensor(0.0, dtype=torch.float32)

        for trajectory in trajectory_batch:
            # reward to go since this is monte carlo form of policy gradient
            G = torch.tensor(0.0, dtype=torch.float32)
            for s, a, r in reversed(trajectory):
                G = self.gamma * G + torch.tensor(r, dtype=torch.float32)
                s_one_hot = F.one_hot(torch.tensor(s), num_classes=self.env.n_states).to(torch.float32)

                log_prob = self.forward_pass(s_one_hot, torch.tensor(a))
                loss += -log_prob * G
        loss /= len(trajectory_batch)

        self.backward_pass(loss)


    def compute_reward_to_go(self, trajectory, gamma):
        """Return list of reward-to-go values for each step in a trajectory."""
        returns = []
        G = 0.0
        for _, _, r in reversed(trajectory):
            G = r + gamma * G
            returns.insert(0, G)  # prepend
        return returns
    
    def forward_pass(self, s, a):
        logits = self.pi(s)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(a)
        return log_prob
    
    def backward_pass(self, loss):
        self.pi_optimizer.zero_grad()
        loss.backward()
        self.pi_optimizer.step()

    def update_policy_batch(self, trajectory_batch: list[list[tuple[int, int, float]]]):
        
        loss = torch.tensor(0.0, dtype=torch.float32)

        all_s =torch.stack(
            [F.one_hot(torch.tensor(s), num_classes=self.env.n_states).to(torch.float32)
                for trajectory in trajectory_batch for s,_,_ in trajectory]
        )

        all_a = torch.tensor([a for trajectory in trajectory_batch for _,a,_ in trajectory])
        all_G = torch.tensor([G for trajectory in trajectory_batch 
                             for G in self.compute_reward_to_go(trajectory, self.gamma)])
        
        all_log_probs = self.forward_pass(all_s, all_a)

        loss = -(all_log_probs * all_G).mean()

        self.backward_pass(loss)


    def fit(self, num_iterations=100, batch_size=100):
        
        best_avg_gamma_return = float('-inf')
        best_pi = None
        for i in tqdm(range(num_iterations), desc="Training", leave=True):
            trajectory_batch = []
            for _ in range(batch_size):
                trajectory_batch.append(self.env.sample_trajectory_pi_nn(self.pi, max_steps=200))
            self.update_policy_batch(trajectory_batch)

            if i % 1 == 0:  # Update every 1 iterations for better performance
                metrics_dict = self.env.evaluate_policy_metrics(num_episodes=200, gamma=self.gamma, pi=self.pi)

                print(f"metrics_dict at i: {i} is {metrics_dict}")
                self.logger["success_rates"].append(metrics_dict["success_rate"])
                avg_gamma_return = metrics_dict["avg_gamma_return"]
                self.logger["gamma_returns"].append(avg_gamma_return)
                self.logger["eval_avg_traj_lengths"].append(metrics_dict.get("avg_trajectory_length", None) if metrics_dict.get("avg_trajectory_length", None) is not None else -1.0)

                if avg_gamma_return > best_avg_gamma_return:
                    # best_pi = self.pi.copy()
                    best_policy_iteration = i
                    best_avg_gamma_return = avg_gamma_return
                
                # Plot real-time metrics
                self._plot_realtime_metrics()
        
        # Store the best policy iteration for plotting
        self.logger["best_policy_iteration"] = best_policy_iteration
        
        # Turn off interactive mode and keep the plot open
        if self.fig is not None:
            plt.ioff()  # Turn off interactive mode
            plt.show(block=True)  # Keep the plot open until manually closed

    def _plot_realtime_metrics(self):
        """Plot real-time training metrics in the same window."""
        import matplotlib.pyplot as plt
        
        if len(self.logger["success_rates"]) > 0:
            success_rates = self.logger["success_rates"]
            gamma_returns = self.logger["gamma_returns"]
            iterations = [i for i in range(len(success_rates))]  # Since we update every iteration now
            
            # Initialize plot window on first call
            if self.fig is None:
                plt.ion()  # Turn on interactive mode
                self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
                self.ax2 = self.ax1.twinx()
                plt.show(block=False)
            
            # Clear the axes
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot success rate
            line1 = self.ax1.plot(iterations, success_rates, 'b-', label='Success Rate', linewidth=2)
            self.ax1.set_xlabel('Iteration')
            self.ax1.set_ylabel('Success Rate', color='b')
            self.ax1.tick_params(axis='y', labelcolor='b')
            self.ax1.set_ylim(0, 1)  # Success rate is between 0 and 1
            
            # Plot gamma return
            line2 = self.ax2.plot(iterations, gamma_returns, 'g-', label='Gamma Return', linewidth=2)
            self.ax2.yaxis.set_label_position('right')  # Ensure label is on right side
            self.ax2.set_ylabel('Gamma Return', color='g')
            self.ax2.tick_params(axis='y', labelcolor='g')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            self.ax1.legend(lines, labels, loc='upper left')
            
            self.ax1.set_title('Policy Training Progress (Real-time)')
            self.ax1.grid(True, alpha=0.3)
            
            # Update the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to update the plot

    def plot_training_stats(self):
        """Plot final training statistics."""
        self._plot_realtime_metrics()
        plt.title('Policy Training Progress (Final)')
        plt.show()  # Show final plot with blocking


if __name__ == "__main__":
    env = ModelFreeFrozenLake(
        step_penalty=-0.15, 
        hole_penalty=-5.0,
        goal_reward=20.0,
    )

    policy_gradient = PolicyGradient(
        env,
        gamma=0.99,
        step_size=0.001,
        hidden=20,
    )
    start_time = time.time()
    metrics = policy_gradient.fit(num_iterations=1000, batch_size=100)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

