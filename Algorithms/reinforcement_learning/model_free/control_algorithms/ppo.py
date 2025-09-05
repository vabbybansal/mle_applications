## PPO (Proximal Policy Optimization) with shared layers for actor and critic
## General outline:
## For batch in (Collect m trajectory's n_steps in batch)
####### Freeze old policy and value function using the batch
####### For x epochs and y minibatches:
############ Calculate new Policy and Value
############ Update Shared Policy and Value

import sys
import os
from ray.util.actor_pool import ActorPool

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FREE_DIR = os.path.dirname(THIS_DIR)  # .../model_free
# (Optional) make driver also robust:
if MODEL_FREE_DIR not in sys.path:
    sys.path.insert(0, MODEL_FREE_DIR)


import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problems.BaseRLEnvironment import BaseRLEnvironment
from problems.model_free_frozen_lake import ModelFreeFrozenLake
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import ray

# Actor Critic
# Baseline is the value function V(s) which estimates the expected return from state s
# For V target in value function learning, we use the TD target: r + gamma * V(s_dash)

# TODOs
# 1) update every multiple steps instead of a single step in trajectory - DONE


@ray.remote(num_cpus=1)
class RemoteTrajectoryWorker:
    def __init__(self, env_kwargs, num_states, num_actions, hidden_dim, model_free_dir):

        import os, sys
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        import torch  # import AFTER env vars
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from torch import nn

        # import custom packages
        if model_free_dir not in sys.path:          # Ensure the worker can import "problems.*"
            sys.path.insert(0, model_free_dir)
        from problems.model_free_frozen_lake import ModelFreeFrozenLake  # now resolves

        # Load Environment
        self.env = ModelFreeFrozenLake(**env_kwargs)

        # Load Policy
        self.actorcriticnet = JointActorCriticNet(num_states, num_actions, hidden_dim)

    
    def set_weights(self, weights_ref: dict):
        # Load Policy Weights
        self.actorcriticnet.load_state_dict(weights_ref)  # Load policy weights from main process
        self.actorcriticnet.eval()  # Set to evaluation mode

    def take_n_steps(self, num_steps: int, s: int, traj_idx: int):
        # Sample n steps, stopping early if done=True
        n_steps = []
        current_state = s
        
        for i in range(num_steps):
            with torch.no_grad():
                s, a, r, s_dash, done = self.env.sample_trajectory_step_sars_pi_nn(self.actorcriticnet, current_state)
                n_steps.append((s, a, r, s_dash, done))
                
                if done:
                    break  # Stop if episode ends
                
                current_state = s_dash  # Continue from next state
                
        return n_steps, traj_idx


class Step:
    def __init__(self, s: int, a: int, r: float, s_dash: int, done: bool):
        self.s = s
        self.a = a
        self.r = r
        self.s_dash = s_dash
        self.done = done

class Trajectory:
    def __init__(self):
        self.steps = []
        self.done = False    

class TrajectoryManager:
    
    def __init__(self, batch_size: int, env: BaseRLEnvironment, actorcriticnet: nn.Module, num_cpus: int, num_steps_per_update: int):
        self.env = env
        self.actorcriticnet = actorcriticnet
        self.batch_size = batch_size
        self.num_steps_per_update = num_steps_per_update

        self.net_hidden_dim = self.actorcriticnet.shared_base[0].out_features

        self.env_kwargs = {
            "step_penalty": self.env.step_penalty,
            "hole_penalty": self.env.hole_penalty,
            "goal_reward": self.env.goal_reward,
        }
        
        self.active_trajectories = [Trajectory() for i in range(batch_size)]
        self.trajectory_history = [] # stores completed trajectories for replay

        self.workers = [RemoteTrajectoryWorker.remote(self.env_kwargs, self.env.n_states, self.env.n_actions, self.net_hidden_dim, MODEL_FREE_DIR) for _ in range(num_cpus)]
        self.pool = ActorPool(self.workers)


    def step_forward(self, i, sync_every):
        '''
        Step forward all active trajectories.
        If a trajectory is done, add it to the trajectory history and replace it with a new trajectory.
        '''

        # Sync weights with workers periodically. Less frequent syncing means workers use slightly outdated policies but improves performance
        if i % sync_every == 0:  # sync weights
            weights_cpu = {k: v.detach().cpu() for k, v in self.actorcriticnet.state_dict().items()}
            ref = ray.put(weights_cpu)
            futures = []
            for w in self.workers:
                futures.append(w.set_weights.remote(ref))
            ray.get(futures)  # wait for all workers to sync
            # Now all workers have the latest policy weights

        jobs = []
        for idx, traj in enumerate(self.active_trajectories):
            s0 = traj.steps[-1].s_dash if traj.steps else 0
            jobs.append((self.num_steps_per_update, s0, idx))  # (num_steps, start_state, traj_id)

        # submit all jobs; pool will queue them across 6 workers
        for args in jobs:
            self.pool.submit(lambda a, v: a.take_n_steps.remote(*v), args)

        # collect results (completion order)
        results_batch = [self.pool.get_next() for _ in range(len(jobs))]

        out_batch = [] # (s,a,r,s_dash,done)

        # Process results from workers
        for result in results_batch:
            n_steps, traj_idx = result

            # Process each step in the n_steps sequence
            for step in n_steps:
                s, a, r, s_dash, done = step

                # add step to trajectory
                self.active_trajectories[traj_idx].steps.append(Step(s, a, r, s_dash, done))
                self.active_trajectories[traj_idx].done = done

                # Add to output batch
                out_batch.append((s, a, r, s_dash, 1 if done else 0))

                if done:
                    self.active_trajectories[traj_idx] = Trajectory()
                    break  # Stop processing more steps for this trajectory

        return out_batch


class JointActorCriticNet(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=128):
        super(JointActorCriticNet, self).__init__()
        self.shared_base = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_dim, num_actions)  # outputs logits per action
        self.critic = nn.Linear(hidden_dim, 1) # outputs value for the input state s
    
    def forward(self, x):
        shared = self.shared_base(x)
        pi = self.actor(shared)
        v = self.critic(shared).squeeze(-1)
        return pi, v


class PPO:
    def __init__(
        self,
        env: BaseRLEnvironment,
        gamma: float = 0.99,
        pi_step_size: float = 3e-4,
        v_step_size: float = 1e-3,
        hidden: int = 20,
        clip_ratio: float = 0.15,
        ent_coef: float = 0.01,
        v_loss_coef: float = 0.5,
    ):
        self.env = env

        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.v_loss_coef = v_loss_coef

        self.actor_critic_net = JointActorCriticNet(env.n_states, env.n_actions, hidden)
        self.actor_critic_net_old = JointActorCriticNet(env.n_states, env.n_actions, hidden)

        # initilize different optimizer param groups since we want different learning rates for actor, critic and shared base
        self.actor_critic_net_optimizer = optim.Adam([
            {'params': self.actor_critic_net.shared_base.parameters(), 'lr': pi_step_size},
            {'params': self.actor_critic_net.actor.parameters(), 'lr': pi_step_size},
            {'params': self.actor_critic_net.critic.parameters(), 'lr': v_step_size},
        ])

        self.gamma = gamma
        self.logger = {"success_rates": [], "gamma_returns": [], "eval_avg_traj_lengths": []}
        
        # Initialize plot for real-time updates
        self.fig = None
        self.ax1 = None
        self.ax2 = None

        # Init Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True, 
                include_dashboard=False,
                num_cpus=6  # Adjust based on your CPU cores
            )

    def clip(self, ratio, clip_ratio):
        return torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    def backward_pass(self, loss, network, optimizer, clipping=0.5):
        optimizer.zero_grad()
        loss.backward()
        # clip grads
        torch.nn.utils.clip_grad_norm_(network.parameters(), clipping)
        optimizer.step()

    def prepare_PPO_batch(self, step_batch: list):

        all_s = torch.stack(
            [F.one_hot(torch.tensor(s), num_classes=self.env.n_states).to(torch.float32)
                for s,_,_,_,_ in step_batch]
        )
        all_a = torch.tensor([a for _,a,_,_,_ in step_batch])
        all_s_dash = torch.stack(
            [F.one_hot(torch.tensor(s_dash), num_classes=self.env.n_states).to(torch.float32)
                for _,_,_,s_dash,_ in step_batch]
        )
        all_r = torch.tensor([r for _,_,r,_,_ in step_batch], dtype=torch.float32)
        all_done = torch.tensor([done for _,_,_,_,done in step_batch], dtype=torch.float32)

        # Fix old policy logits and v
        with torch.no_grad():
            pi_logits_old, v_logits_old = self.actor_critic_net_old(all_s)
            _, v_dash_logits = self.actor_critic_net_old(all_s_dash)
            dist_old = torch.distributions.Categorical(logits=pi_logits_old)
            log_p_old = dist_old.log_prob(all_a).detach()

            v_td_target = (all_r + self.gamma * v_dash_logits * (1 - all_done)).detach()
            A = (v_td_target - v_logits_old).detach()
            A = (A - A.mean()) / (A.std() + 1e-8)

        return dict(all_s=all_s, all_a=all_a, A=A, log_p_old=log_p_old, v_td_target=v_td_target)



    def policy_step(self, cached_batch: dict, epochs: int, minibatch_size: int):
        '''
        Update actor policy pi and critic value function v with the collected trajectories
        '''
        N = cached_batch['all_s'].shape[0]
        for ep in range(epochs):
            idx = torch.randperm(N)
            for start in range(0, N, minibatch_size):
                mb_idx = idx[start:start+minibatch_size]
                all_s, all_a = cached_batch['all_s'][mb_idx], cached_batch['all_a'][mb_idx]
                A_frozen = cached_batch['A'][mb_idx]
                V_td_target_frozen = cached_batch['v_td_target'][mb_idx]
                log_p_old = cached_batch['log_p_old'][mb_idx]

                # New policy and value on this mini batch
                pi_logits_new, v_new = self.actor_critic_net(all_s)
                dist_new = torch.distributions.Categorical(logits=pi_logits_new)
                log_p_new = dist_new.log_prob(all_a)

                # ratios 
                ratio = torch.exp(log_p_new - log_p_old)
                clipped_ratio = self.clip(ratio, self.clip_ratio)
                policy_loss = -torch.min(ratio * A_frozen, clipped_ratio * A_frozen).mean()

                # value and entropy
                value_loss = F.smooth_l1_loss(v_new, V_td_target_frozen)
                entropy_loss = dist_new.entropy().mean()

                loss = policy_loss + self.v_loss_coef * value_loss - self.ent_coef * entropy_loss

                # Optimizer step for updating the policy and value function
                self.backward_pass(loss, self.actor_critic_net, self.actor_critic_net_optimizer, clipping=0.5)

            # optional TODO: early stop if batch KL drifts too far

    def fit(self, num_iterations=100, batch_size=100, num_steps_per_update=50, plot_every=10, sync_every=1, epochs=10, minibatch_size=32):

        best_avg_gamma_return = float('-inf')
        best_pi = None

        trajectory_manager = TrajectoryManager(batch_size, self.env, self.actor_critic_net, num_cpus=6, num_steps_per_update=num_steps_per_update)

        for i in tqdm(range(num_iterations), desc="Training", leave=True):
            steps_batch = trajectory_manager.step_forward(i, sync_every)  # (s,a,r,s_dash)

            self.actor_critic_net_old.load_state_dict(self.actor_critic_net.state_dict())
            self.actor_critic_net_old.eval()

            batch = self.prepare_PPO_batch(steps_batch)
            # Update policy & value function using collected trajectories
            # self.policy_step(batch, epochs=epochs, mini_batches=mini_batches)
            self.policy_step(batch, epochs=epochs, minibatch_size= minibatch_size)


            # evaluation + logging + plotting
            if i % plot_every == 0:
                metrics_dict = self.env.evaluate_policy_metrics(num_episodes=200, gamma=self.gamma, pi=self.actor_critic_net)
                # print(f"metrics_dict at i: {i} is {metrics_dict}")
                self.logger["success_rates"].append(metrics_dict["success_rate"])
                avg_gamma_return = metrics_dict["avg_gamma_return"]
                self.logger["gamma_returns"].append(avg_gamma_return)
                self.logger["eval_avg_traj_lengths"].append(
                    metrics_dict.get("avg_trajectory_length", None)
                    if metrics_dict.get("avg_trajectory_length", None) is not None else -1.0
                )
                if avg_gamma_return > best_avg_gamma_return:
                    best_policy_iteration = i
                    best_avg_gamma_return = avg_gamma_return
                self._plot_realtime_metrics(plot_every)

        self.logger["best_policy_iteration"] = best_policy_iteration

        # Clean up matplotlib
        if self.fig is not None:
            plt.ioff()
            plt.show(block=True)

        # Ensure Ray is shutdown properly
        try:
            ray.shutdown()
        except Exception:
            pass

    def _plot_realtime_metrics(self, plot_every):
        """Plot real-time training metrics in the same window."""
        import matplotlib.pyplot as plt
        
        if len(self.logger["success_rates"]) > 0:
            success_rates = self.logger["success_rates"]
            gamma_returns = self.logger["gamma_returns"]
            iterations = [plot_every * i for i in range(len(success_rates))]  # Scale x-axis by plot_every
            
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
            
            self.ax1.set_title('Policy Training Progress (Real-time) v2')
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

    ppo = PPO(
        env,
        gamma=0.99,
        pi_step_size=3e-4,
        v_step_size=1e-3,
        hidden=20,
        clip_ratio=0.15,
    )
    start_time = time.time()
    metrics = ppo.fit(
        num_iterations=10000, 
        batch_size=6,  # these many trajectories in parallel
        num_steps_per_update=50,  # these many steps per update
        plot_every=50,
        sync_every=1, # importance sampling only corrects the prob difference between the old and new policy for the same states, hence we should sync at every step for PPO. We should have mutliple mini batches per samples collected though.
        epochs=10,
        minibatch_size=32,
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

