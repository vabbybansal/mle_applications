from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problems.BaseRLEnvironment import BaseRLEnvironment
from problems.model_free_frozen_lake import ModelFreeFrozenLake



# Cost Function => J(w), parameterized by w
# J(w) => MSE => 1/2*[Q_pi(s,a) - Q_pi_hat(s,a,w)]**2
# delta_J(w) = [Q_pi(s,a) - Q_pi_hat(s,a,w)] * delta_Q_pi_hat(s,a,w) {with respect to w}
# update for w: w_k+1 = w_k - ⍺ * delta_J(w) {⍺ is learning rate}
# => w_k+1 = w_k - ⍺ * [Q_pi(s,a) - Q_pi_hat(s,a,w)] * delta_Q_pi_hat(s,a,w)
# But do we really have a supervisor? sort of => this is the reward from the sampled trajectories

# For on policy SARSA, TD target is r + γ * Q_pi(s_dash, a_dash, w_k)
# For TD(0), supervisor is the immediate reward + bootstraped value from the next state
# => w_k+1 = w_k - ⍺ * [r + γ * Q_pi(s_dash, a_dash, w_k) - Q_pi_hat(s,a,w_k)] * delta_Q_pi_hat(s,a,w_k)

# For off policy Q Learning, TD target is r + γ * max_a[Q_pi(s_dash, a_dash, w_k)]
# For TD(0), supervisor is the immediate reward + bootstraped value from the next state
# => w_k+1 = w_k - ⍺ * [r + γ * max_a[Q_pi(s_dash, a_dash, w_k)] - Q_pi_hat(s,a,w_k)] * delta_Q_pi_hat(s,a,w_k)

# Implementation
# We init Q_hat with a neural network with weights w_0
# We want to push Q_hat towards optimum policy Q_pi
# Min this MSE Loss => argmin_w (Q_pi - Q_hat)**2
# Q_pi is our supervisor. For Q Learning, this would be equal to the TD target (r + γ * max_a[Q_pi(s_dash, a_dash, w_k)])
# Q_pi(s_dash, a_dash, w_k) is not included in the differentiation - partial gradients.
# Now, for each TD step, we sample r,s',a' and update w using gradient descent using autograd,
# which internally will compute w_k+1 = w_k - ⍺ * [r + γ * max_a[Q_pi(s_dash, a_dash, w_k)] - Q_pi_hat(s,a,w_k)] * delta_Q_pi_hat(s,a,w_k)


'''
input: state s
output: Q(s,a), one for each action
'''
class QNet(nn.Module):

    def __init__(self, n_states, n_actions, hidden=20):
        super().__init__()

        # input: state s, output: Q(s,a), one for each action
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),nn.ReLU(),nn.Linear(hidden, n_actions),)
    def forward(self, x):
        return self.net(x)


class Q_Learning_Function_Approximation:
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
        step_size: float = 0.001,
        step_size_decay: float = 0.995,
        hidden: int = 20,
    ):
        self.env = env
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.step_size = step_size
        self.step_size_decay = step_size_decay
        # init neural network with action, state space
        self.q_net = QNet(env.n_states, env.n_actions, hidden)

        # How to decay lr 
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=step_size)

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
            "num_times_goal_touched": 0,
        }

    def epsilon_greedy_improvement(self):
        # decay epsilon (GLIE)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        # decay alpha
        self.alpha *= self.alpha_decay
        self.alpha = max(self.alpha, self.alpha_min)
        # update the policy using epsilon greedy (Note: this is not needed since for sampling, we directly use Q table)
        # self.env.policy = self.epsilon_greedy_policy_update(self.env.Q, self.epsilon)

        # decay step size
        self.step_size *= self.step_size_decay
        for g in self.optimizer.param_groups:
            g['lr'] = self.step_size

    def sample_epsilon_greedy_action(self, s:int, epsilon: float) -> tuple[int, bool]:
        '''
        input: state s, epsilon (float)
        output: action a, is_sample_exploration (bool)

        Greedy: finds the best action with the highest Q value in the Q NN network
        '''

        # if epsilon, choose an action randomly
        if np.random.rand() < epsilon:
            return np.random.randint(self.env.n_actions), True
        else:
            # choose greedily
            state_tensor = torch.nn.functional.one_hot(torch.tensor(s), num_classes=self.env.n_states).to(torch.float32).unsqueeze(0)  # shape: (batch, 1)
            # get q values from the Q network
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                # index of the best action
                best_action = torch.argmax(q_values, dim=1).item()
                return best_action, False

    def evaluation(self, max_steps: int = 200):
        
        G = 0.0
        done = False

        s, steps = 0, 0
        

        trajectory = []
        traj_explored = False


        # w_k+1 = w_k - ⍺ * [r + γ * max_a[Q_pi(s_dash, a_dash, w_k)] - Q_pi_hat(s,a,w_k)] * delta_Q_pi_hat(s,a,w_k)
        while not done and steps < max_steps:

            # Choose a from s using policy derived from Q (eg. epsilon greedy)
            a, is_sample_exploration = self.sample_epsilon_greedy_action(s, self.epsilon)
            traj_explored = traj_explored or is_sample_exploration

            # Take action a and observe response r and s'
            r, s_dash, done =  self.env.step_s_a(s, a)

            if s_dash == 63:
                self.logger["num_times_goal_touched"] += 1

            # log traj
            trajectory.append((s, a, r))

            # convert r to torch tensor
            r_tensor = torch.tensor([r], dtype=torch.float32).unsqueeze(0)

            if done:
                td_target = r_tensor
            else:
                with torch.no_grad(): # partial gradient for this update - standard practice
                    s_dash_tensor = torch.nn.functional.one_hot(torch.tensor(s_dash), num_classes=self.env.n_states).to(torch.float32).unsqueeze(0)
                    # TD Target for Q Learning = r + gamma * max_a[Q_pi(s_dash, a_dash, w_k)]
                    td_target = r_tensor + self.gamma * self.q_net(s_dash_tensor).max(dim=1).values

            s_tensor = torch.nn.functional.one_hot(torch.tensor(s), num_classes=self.env.n_states).to(torch.float32).unsqueeze(0)
            q_sa = self.q_net(s_tensor).gather(dim=1, index=torch.tensor([a], dtype=torch.long).unsqueeze(0))

            td_target = td_target.view(-1,1)
            q_sa = q_sa.view(-1,1)

            # loss  = torch.nn.functional.mse_loss(td_target, q_sa)
            loss  = torch.nn.functional.smooth_l1_loss(td_target, q_sa)
            
            # Backpropagate the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            G += r * (self.gamma ** steps)
            steps += 1

            if not done:
                s = s_dash

        self.logger["trajectories"].append(trajectory)
        self.logger["trajectory_returns"].append(round(G, 2))
        self.logger["trajectory_lengths"].append(steps)
        self.logger["exploration_flags"].append(traj_explored)

        

    def snapshot_Q(self):
        Q = np.zeros((self.env.n_states, self.env.n_actions), dtype=np.float32)
        with torch.no_grad():
            for s in range(self.env.n_states):
                x = torch.nn.functional.one_hot(torch.tensor(s), num_classes=self.env.n_states).float().unsqueeze(0)
                Q[s] = self.q_net(x).squeeze(0).cpu().numpy()
        return Q

        

    def fit(self, num_iterations=1000) -> dict:
        best_gamma_return = float('-inf')
        best_success = -1
        best_Q = None
        best_policy_iteration = None
        best_avg_gamma_return = None
        metrics = {}

        for i in tqdm(range(num_iterations), desc="Training", leave=True):

            self.logger["epsilon_values"].append(self.epsilon)  # Log epsilon
            self.logger["alpha_values"].append(self.alpha)      # Log alpha (even if decaying)

            # evaluate trajectory (also logs return)
            self.evaluation()

            # improve policy and decay epsilon/alpha
            self.epsilon_greedy_improvement()

            

            if i % 100 != 0:
                # log -1.0 for all intermediate iterations
                self.logger["success_rates"].append(-1.0)
                self.logger["gamma_returns"].append(-1.0)
                self.logger["eval_avg_traj_lengths"].append(-1.0)

            # policy checkpoint
            if i % 100 == 0:  # evaluate every 100 episodes
                self.env.Q = self.snapshot_Q()
                metrics_dict = self.env.evaluate_policy_metrics(num_episodes=200, gamma=self.gamma)
                success = metrics_dict["success_rate"]
                avg_gamma_return = metrics_dict["avg_gamma_return"]
                avg_traj_len_eval = metrics_dict.get("avg_trajectory_length", None)
                self.logger["success_rates"].append(success)
                self.logger["gamma_returns"].append(avg_gamma_return)
                self.logger["eval_avg_traj_lengths"].append(avg_traj_len_eval if avg_traj_len_eval is not None else -1.0)
                
                if avg_gamma_return > best_gamma_return:
                    best_gamma_return = avg_gamma_return
                    best_Q = self.snapshot_Q()
                    best_policy_iteration = i
                    best_avg_gamma_return = avg_gamma_return
                if success > best_success:
                    best_success = success

        # Store the best policy iteration for plotting
        self.logger["best_policy_iteration"] = best_policy_iteration
        
        if best_Q is not None:
            self.env.Q = best_Q

        self.env.Q = self.snapshot_Q()

        final_metrics = self.env.evaluate_policy_metrics(num_episodes=200, gamma=self.gamma)
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

if __name__ == "__main__":
    env = ModelFreeFrozenLake(
        # very interesting behavior if this value is -0.01. At cell 47, since there are two H adjacent 
        # to the main paths below, taking down action would actually lead to lower expected reward since agent can 
        # fall to the hole (left) with 10 percent probs. At the same time, taking right action at cell 47 and below 
        # makes sure that the agent does not fall in the holes at all and they can still the goal with a 10% probability.
        # With a lower step penalty of -0.01, the agent has incentive to hit the wall again and again and it gets stuck 
        # at 47 as per greedy policy. Making this -0.15, the incentive to increase path length decreases (and self cell looping),
        # hence it has incentive to find the right path to the goal.
        step_penalty=-0.0015, 
        hole_penalty=-5.0,
        goal_reward=20.0,
    )
    q_Learning_Function_Approximation = Q_Learning_Function_Approximation(
        env,
        epsilon=0.9,
        epsilon_decay=0.9999,
        epsilon_min=0.00001,
        gamma=0.99,
        alpha=0.5,
        alpha_decay=0.999,
        alpha_min=0.001,
        hidden=50,
    )
    start_time = time.time()
    metrics = q_Learning_Function_Approximation.fit(num_iterations=10000)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("**************************************************")
    print("Metrics: ", metrics)
    print("**************************************************")
    # print(q_Learning_Function_Approximation.env.Q)

    print(f"Number of times goal was touched: {q_Learning_Function_Approximation.logger['num_times_goal_touched']}")    

    env.plot_training_stats(q_Learning_Function_Approximation.logger, smooth_window=50)
    env.plot_q_heatmap(title="Q-Value Heatmap (final Q)", show=False)
    plt.title("Q-Learning w Function Approximation")
    plt.show()

