import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BaseRLEnvironment import BaseRLEnvironment

class ModelFreeFrozenLake(gym.Env, BaseRLEnvironment):
    def __init__(self, 
                 map_name="8x8", 
                 is_slippery=True,
                 step_penalty=-0.01,
                 hole_penalty=-5.0,
                 goal_reward=10.0,
                 Q_init_std=1e-4,
        ):
        self.step_penalty = step_penalty
        self.hole_penalty = hole_penalty
        self.goal_reward = goal_reward

        self.env = gym.make(
            "FrozenLake-v1", 
            is_slippery=is_slippery, # stochastic - the agent can slip and land in a neighboring tile even when it decided to move in a different one
            map_name=map_name
        ) 
        self.actual_env = self.env.unwrapped
        self.base_P = self.actual_env.P

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # initialize the policy to a uniform random stochastic policy
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions

        # initialize the value function to 0 for all states
        self.V = np.zeros(self.n_states) # value function is a vector of length n_states, where each element is a scalar
        # self.Q = np.zeros((self.n_states, self.n_actions)) # action-value function is a matrix of shape (n_states, n_actions), where each element is a scalar
        # when creating Q, initialize it to a small random value
        self.Q = np.random.randn(self.n_states, self.n_actions) * Q_init_std

        # modify slip probabilities to realistic values
        self.set_slip_probabilities(0.8, 0.1, 0.1)
        # modify rewards structure to include the step penalty, hole penalty, and goal reward
        self.modify_rewards(self.actual_env)
    
    def __str__(self):
        desc = self.env.unwrapped.desc
        return '\n'.join(' '.join(c.decode("utf-8") for c in row) for row in desc)
    
    def print_transition_probabilities(self, env):
        for state in range(env.nrow * env.ncol):
            for action in range(self.n_actions):
                print(state, action, env.P[state][action])


    def modify_rewards(self, env):
        for state in range(env.nrow * env.ncol):
            for action in range(self.n_actions):
                new_transitions = []
                # This accesses the transition probability table. For a given (state, action), Gym returns a list of possible transitions, each a tuple:
                # (probability, next_state, reward, done)
                for prob, next_state, _, done in env.P[state][action]:
                    # print(state, action, prob, next_state, _, done)
                    row, col = divmod(next_state, env.ncol)
                    tile = env.desc[row][col].decode("utf-8")
                    if tile == "H":
                        reward = self.hole_penalty
                    elif tile == "G":
                        reward = self.goal_reward
                    else:
                        reward = self.step_penalty  # small step penalty
                    new_transitions.append((prob, next_state, reward, done))
                env.P[state][action] = new_transitions

    def set_slip_probabilities(self, p_intended, p_left, p_right):
        """
        In-place override of self.actual_env.P[s][a]:
        • p_intended → your chosen direction,
        • p_left     → rotate action anticlockwise,
        • p_right    → rotate action clockwise.

        p_intended + p_left + p_right must = 1.
        """
        assert abs(p_intended + p_left + p_right - 1.0) < 1e-6

        P = self.actual_env.P
        ncol = self.actual_env.ncol
        nrow = self.actual_env.nrow

        # grid‐based deltas for actions: 0=LEFT,1=DOWN,2=RIGHT,3=UP
        deltas = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}

        def reward(ns):
            r, c = divmod(ns, ncol)
            t = self.actual_env.desc[r][c].decode()
            if t=='H': return self.hole_penalty
            if t=='G': return self.goal_reward
            return self.step_penalty
        
        def done_for(ns):
            r, c = divmod(ns, ncol)
            return self.actual_env.desc[r][c].decode() in ('H', 'G')

        for s in range(self.n_states):
            r, c = divmod(s, ncol)
            for a in range(self.n_actions):
                # intended move
                dr, dc = deltas[a]
                ri, ci = min(max(r+dr,0),nrow-1), min(max(c+dc,0),ncol-1)
                ns_i = ri * ncol + ci

                # slip‐left  = (a-1)%4
                al = (a-1) % 4
                dr, dc = deltas[al]
                rl, cl = min(max(r+dr,0),nrow-1), min(max(c+dc,0),ncol-1)
                ns_l = rl * ncol + cl

                # slip‐right = (a+1)%4
                ar = (a+1) % 4
                dr, dc = deltas[ar]
                rr, cr = min(max(r+dr,0),nrow-1), min(max(c+dc,0),ncol-1)
                ns_r = rr * ncol + cr

                # overwrite transitions
                P[s][a] = [
                    (p_intended, ns_i, reward(ns_i), done_for(ns_i)),
                    (p_left,     ns_l, reward(ns_l), done_for(ns_l)),
                    (p_right,    ns_r, reward(ns_r), done_for(ns_r)),
                ]

    

    def sample_trajectory(self, epsilon_greedy: bool = True, epsilon: float = 0.1, max_steps: int = 10000) -> tuple[list[tuple[int, int, float]], bool]:
        """
        Samples a trajectory from the environment.

        Returns:
            list of (state, action, reward) tuples:
                state  (int): the state index
                action (int): the action taken
                reward (float): the reward received after taking the action
        """

        trajectory = []
        state = 0
        steps = 0
        is_sample_exploration = False

        while True:

            # the policy is epsilon soft
            # action = np.random.choice(self.n_actions, p=self.policy[state])
            if epsilon_greedy:

                if np.random.rand() < epsilon:
                    action = np.random.randint(self.n_actions)
                    is_sample_exploration = True
                else:
                    # action = np.argmax(self.Q[state])
                    # choose greedily with random tie-breaking
                    best_actions = np.flatnonzero(self.Q[state] == self.Q[state].max())
                    action = np.random.choice(best_actions)
            else:
                action = np.argmax(self.Q[state])


            transitions = self.actual_env.P[state][action]
            probs = [t[0] for t in transitions]
            next_states = [t[1] for t in transitions]
            rewards = [t[2] for t in transitions]
            dones = [t[3] for t in transitions]

            idx = np.random.choice(len(transitions), p=probs)
            next_state = next_states[idx]            
            reward = rewards[idx]
            done = dones[idx]

            trajectory.append((state, action, reward))
            state = next_state
            steps += 1

            if done or steps >= max_steps:
                break

        return trajectory, is_sample_exploration
    
    
    def print_greedy_path_grid(self, ax=None, max_steps=200):
        """
        Plots the greedy path on the grid using matplotlib if ax is provided,
        otherwise prints an ASCII grid to the console.
        """
        env = self.env.unwrapped
        nrow, ncol = env.nrow, env.ncol
        desc = env.desc
        greedy = self.Q.argmax(axis=1)

        # rollout from Start, deterministic next cell (ignoring slip)
        s = 0
        visited = set([s])
        for _ in range(max_steps):
            r, c = divmod(s, ncol)
            tile = desc[r][c].decode()
            if tile in ('H','G'):  # stop at terminal
                break
            a = greedy[s]
            nr, nc = r - (a==3) + (a==1), c - (a==0) + (a==2)  # UP/DOWN, LEFT/RIGHT
            nr = min(max(nr,0), nrow-1); nc = min(max(nc,0), ncol-1)
            ns = nr*ncol + nc
            if ns == s or ns in visited:  # loop or stuck
                visited.add(ns)
                break
            visited.add(ns)
            s = ns

        # Build grid for visualization
        grid = np.full((nrow, ncol), '-', dtype='<U2')
        for i in range(nrow):
            for j in range(ncol):
                idx = i*ncol + j
                ch = desc[i][j].decode()
                if ch in ('S','H','G'):
                    grid[i, j] = ch
                elif idx in visited:
                    grid[i, j] = 'o'

        if ax is not None:
            # Plot the grid using matplotlib
            cmap = plt.get_cmap("Blues")
            ax.imshow(np.zeros_like(grid, dtype=float), cmap=cmap, vmin=0, vmax=1)
            for i in range(nrow):
                for j in range(ncol):
                    val = grid[i, j]
                    color = "black"
                    if val == 'S':
                        color = "green"
                    elif val == 'G':
                        color = "gold"
                    elif val == 'H':
                        color = "red"
                    elif val == 'o':
                        color = "blue"
                    ax.text(j, i, val, ha='center', va='center', fontsize=16, fontweight='bold', color=color)
            ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
            ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=1, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Greedy Path Grid")
        else:
            # Fallback: print ASCII grid
            rows = [' '.join(grid[i, :]) for i in range(nrow)]
            print('\n'.join(rows))

    def _action_name(self, action):
        # Helper to map action index to name
        names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        return names.get(action, str(action))

    def evaluate_policy_metrics(self, num_episodes: int = 100, gamma: float = 0.99) -> dict:
        """
        Evaluate the current greedy policy (derived from self.Q) over num_episodes.
        Returns a dict with success_rate and avg_gamma_return.
        """
        successes = 0
        returns = []
        for _ in range(num_episodes):
            traj, _ = self.sample_trajectory(epsilon_greedy=False)
            if len(traj) > 0 and traj[-1][2] == self.goal_reward:
                successes += 1
            G = sum(r * (gamma ** t) for t, (_, _, r) in enumerate(traj))
            returns.append(G)
        return {
            "success_rate": successes / num_episodes if num_episodes > 0 else 0.0,
            "avg_gamma_return": float(np.mean(returns)) if len(returns) > 0 else 0.0,
        }

    def success_rate(self, num_episodes: int = 100) -> float:
        """
        Backward-compat helper: returns only success rate of current greedy policy.
        """
        metrics = self.evaluate_policy_metrics(num_episodes=num_episodes, gamma=0.99)
        return metrics["success_rate"]

    def plot_training_stats(self, logger: dict, smooth_window: int = 50):
        """
        Plot training statistics recorded by a control algorithm logger.

        Expected keys in logger:
        - trajectory_returns: list[float]
        - success_rates: list[float] with -1.0 for non-checkpoints
        - gamma_returns: list[float] with -1.0 for non-checkpoints
        - exploration_flags: list[bool]
        - epsilon_values: list[float]
        - alpha_values: list[float]
        - best_policy_iteration: Optional[int]
        """

        x_ret = list(range(len(logger.get("trajectory_returns", []))))
        y_ret = np.array(logger.get("trajectory_returns", []))

        exploration_flags = np.array(logger.get("exploration_flags", []))
        exploration_indices = np.where(exploration_flags)[0] if len(exploration_flags) > 0 else []

        success_rates = np.array(logger.get("success_rates", []))
        sr_indices = [i for i, v in enumerate(success_rates) if v != -1.0]
        sr_values = [success_rates[i] for i in sr_indices]

        gamma_returns = np.array(logger.get("gamma_returns", []))
        gr_indices = [i for i, v in enumerate(gamma_returns) if v != -1.0]
        gr_values = [gamma_returns[i] for i in gr_indices]

        epsilon_values = np.array(logger.get("epsilon_values", []))
        alpha_values = np.array(logger.get("alpha_values", []))

        best_policy_iteration = logger.get("best_policy_iteration", None)

        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1], wspace=0.25, hspace=0.35)

        # Left panel axes
        ax_ret = fig.add_subplot(gs[0, 0])
        ax_sr = fig.add_subplot(gs[1, 0])
        ax_gr = fig.add_subplot(gs[2, 0])

        # Trajectory returns
        if smooth_window > 1 and len(y_ret) >= smooth_window:
            y_ret_smooth = np.convolve(y_ret, np.ones(smooth_window)/smooth_window, mode='valid')
            x_ret_smooth = x_ret[smooth_window-1:]
            ax_ret.plot(x_ret_smooth, y_ret_smooth, label=f'Trajectory Return (smoothed, window={smooth_window})', color='green', linewidth=2, alpha=0.9)
        else:
            ax_ret.plot(x_ret, y_ret, label='Trajectory Return', color='green', linewidth=2, alpha=0.9)

        # Exploration markers
        if len(exploration_indices) > 0:
            for idx in exploration_indices:
                ax_ret.axvline(idx, color='red', linestyle=':', alpha=0.25, linewidth=1)
            from matplotlib.lines import Line2D
            proxy = Line2D([0], [0], color='red', linestyle=':', alpha=0.5, linewidth=2)
            handles, labels = ax_ret.get_legend_handles_labels()
            handles.append(proxy)
            labels.append("Exploration Trajectory")
        else:
            handles, labels = ax_ret.get_legend_handles_labels()

        # Best policy marker
        if best_policy_iteration is not None:
            ax_ret.axvline(best_policy_iteration, color='black', linestyle='-', linewidth=2, alpha=0.8, label="Best Policy")
            from matplotlib.lines import Line2D
            proxy_black = Line2D([0], [0], color='black', linestyle='-', linewidth=2, alpha=0.8)
            handles.append(proxy_black)
            labels.append("Best Policy")

        ax_ret.legend(handles, labels)
        ax_ret.set_xlabel("Iteration")
        ax_ret.set_ylabel("Trajectory Return")
        ax_ret.set_title("Trajectory Returns over Iterations")
        ax_ret.grid(True, linestyle='--', alpha=0.5)

        # Success rate
        if len(sr_indices) > 0:
            ax_sr.plot(sr_indices, sr_values, marker='o', color='blue', label='Success Rate (checkpoint)')
            ax_sr.set_ylim(-0.05, 1.05)
        else:
            ax_sr.plot([], [], label='No Success Rate Data')
        if len(exploration_indices) > 0:
            for idx in exploration_indices:
                ax_sr.axvline(idx, color='red', linestyle=':', alpha=0.25, linewidth=1)
            from matplotlib.lines import Line2D
            proxy = Line2D([0], [0], color='red', linestyle=':', alpha=0.5, linewidth=2)
            handles, labels = ax_sr.get_legend_handles_labels()
            handles.append(proxy)
            labels.append("Exploration Trajectory")
            ax_sr.legend(handles, labels)
        else:
            ax_sr.legend()
        ax_sr.set_xlabel("Iteration")
        ax_sr.set_ylabel("Success Rate")
        ax_sr.set_title("Success Rate over Iterations (evaluated every 100 episodes)")
        ax_sr.grid(True, linestyle='--', alpha=0.5)

        # Gamma return
        if len(gr_indices) > 0:
            ax_gr.plot(gr_indices, gr_values, marker='o', color='purple', label='Avg Gamma Return (checkpoint)')
        else:
            ax_gr.plot([], [], label='No Gamma Return Data')
        if len(exploration_indices) > 0:
            for idx in exploration_indices:
                ax_gr.axvline(idx, color='red', linestyle=':', alpha=0.25, linewidth=1)
            from matplotlib.lines import Line2D
            proxy = Line2D([0], [0], color='red', linestyle=':', alpha=0.5, linewidth=2)
            handles, labels = ax_gr.get_legend_handles_labels()
            handles.append(proxy)
            labels.append("Exploration Trajectory")
            ax_gr.legend(handles, labels)
        else:
            ax_gr.legend()
        ax_gr.set_xlabel("Iteration")
        ax_gr.set_ylabel("Avg Gamma Return")
        ax_gr.set_title("Average Gamma Return over Iterations (evaluated every 100 episodes)")
        ax_gr.grid(True, linestyle='--', alpha=0.5)

        # Right top: epsilon and alpha
        ax_eps_alpha = fig.add_subplot(gs[0, 1])
        if len(epsilon_values) > 0:
            ax_eps_alpha.plot(x_ret, epsilon_values, color='orange', label='Epsilon', linewidth=2, alpha=0.8)
        if len(alpha_values) > 0:
            ax_eps_alpha.plot(x_ret, alpha_values, color='brown', label='Alpha', linewidth=2, alpha=0.8)
        if len(exploration_indices) > 0:
            for idx in exploration_indices:
                ax_eps_alpha.axvline(idx, color='red', linestyle=':', alpha=0.25, linewidth=1)
            from matplotlib.lines import Line2D
            proxy = Line2D([0], [0], color='red', linestyle=':', alpha=0.5, linewidth=2)
            handles, labels = ax_eps_alpha.get_legend_handles_labels()
            handles.append(proxy)
            labels.append("Exploration Trajectory")
            ax_eps_alpha.legend(handles, labels)
        else:
            ax_eps_alpha.legend()
        ax_eps_alpha.set_xlabel("Iteration")
        ax_eps_alpha.set_ylabel("Value")
        ax_eps_alpha.set_title("Epsilon and Alpha over Iterations")
        ax_eps_alpha.grid(True, linestyle='--', alpha=0.5)

        # Right middle: greedy path grid
        ax_grid = fig.add_subplot(gs[1, 1])
        try:
            self.print_greedy_path_grid(ax=ax_grid)
        except TypeError:
            ax_grid.text(0.5, 0.5, "print_greedy_path_grid(ax) not supported", ha='center', va='center', fontsize=12)
            ax_grid.set_title("Greedy Path Grid")
            ax_grid.axis('off')

        # Right bottom placeholder
        ax_empty2 = fig.add_subplot(gs[2, 1])
        ax_empty2.axis('off')
        ax_empty2.set_title("")

        plt.tight_layout()
        # plt.show()

if __name__ == "__main__":
    env = ModelFreeFrozenLake()
    trajectory, _ = env.sample_trajectory(epsilon_greedy=True, epsilon=0.1)
    print(len(trajectory))
    # Example usage: plot Q-value heatmap for each action
    # for a in range(env.n_actions):
    #     env.plot_action_value_heatmap(action=a)
    env.print_greedy_path_grid()