import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


    def sample_trajectory_pi_nn(self, pi: nn.Module, max_steps: int = 10000) -> tuple[list[tuple[int, int, float]], bool]:
        """
        Samples a trajectory from the environment using a policy network.

        Returns:
            list of (state, action, reward) tuples:
                state  (int): the state index
                action (int): the action taken
                reward (float): the reward received after taking the action
        """

        trajectory = []
        state = 0
        steps = 0

        while True:
            
            s_one_hot = F.one_hot(torch.tensor(state), num_classes=self.n_states).to(torch.float32).unsqueeze(0)
            logits = pi(s_one_hot)
            if isinstance(logits, tuple): # if the policy network returns a tuple <pi, v>, get the first element
                logits = logits[0]
            else: # if the policy network returns a single logits, get the logits
                logits = logits

            dist = torch.distributions.Categorical(logits=logits) # Create a discrete prob dist by appying softmax to logits
            # samples based on the prob dist
            action = dist.sample().item()


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

        return trajectory
    

    def sample_trajectory_step_sars_pi_nn(self, pi: nn.Module, s:int = 0) -> tuple[int, int, float, int, bool]:
        """
        Samples a step from the environment using a policy network.

        Returns:
            tuple of (state, action, reward, next_state, done):
                state  (int): the state index
                action (int): the action taken
                reward (float): the reward received after taking the action
                next_state (int): the next state index
        """

        s_one_hot = F.one_hot(torch.tensor(s), num_classes=self.n_states).to(torch.float32).unsqueeze(0)

        # get logits from the policy network
        logits = pi(s_one_hot)
        if isinstance(logits, tuple): # if the policy network returns a tuple <pi, v>, get the first element
            logits = logits[0]
        else: # if the policy network returns a single logits, get the logits
            logits = logits

        dist = torch.distributions.Categorical(logits=logits) # Create a discrete prob dist by appying softmax to logits
        # samples based on the prob dist
        a = dist.sample().item()

        transitions = self.actual_env.P[s][a]
        probs = [t[0] for t in transitions]
        next_states = [t[1] for t in transitions]
        rewards = [t[2] for t in transitions]
        dones = [t[3] for t in transitions]

        idx = np.random.choice(len(transitions), p=probs)
        s_dash = next_states[idx]            
        r = rewards[idx]
        done = dones[idx]
        
        return s, a, r, s_dash, done
    

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
    
    def sample_epsilon_greedy_action(self, s:int, epsilon: float) -> tuple[int, bool]:

        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions), True
        else:
            # choose greedily with random tie-breaking
            best_actions = np.flatnonzero(self.Q[s] == self.Q[s].max())
            return np.random.choice(best_actions), False

    
    def step_s_a(self,s:int, a:int) -> tuple[float, int, bool]:
        """
        Makes an s_a step with s state and a action and observe the reward and the next state
        Input:
            state (int)
            action (int)
        Returns:
            reward (float)
            s_dash (int)
            done (bool)
        """

        transitions = self.actual_env.P[s][a]
        probs = [t[0] for t in transitions]
        next_states = [t[1] for t in transitions]
        rewards = [t[2] for t in transitions]
        dones = [t[3] for t in transitions]

        idx = np.random.choice(len(transitions), p=probs)
        s_dash = next_states[idx]            
        r = rewards[idx]
        done = dones[idx]

        return r, s_dash, done
    
    
    import matplotlib.patches as mpatches

    def old_print_greedy_path_grid(self, ax=None, max_steps: int = 200,
                           highlight_color: str = "#cfefff",  # light blue for visited
                           goal_color: str = "#ffd86b",      # gold-ish for goal visited
                           hole_color: str = "#ffb3b3"):     # light red for hole visited
        """
        Draw a nicer greedy deterministic rollout (ignoring slip) from start.
        - If ax is provided, color visited cells and draw base tile letters on top (no overlapping letters).
        - If ax is None, print an ASCII grid (term visited cells become 'G(o)' or 'H(o)').
        """
        import matplotlib.patches as mpatches  # ensure patches available

        env = self.env.unwrapped
        nrow, ncol = env.nrow, env.ncol
        desc = env.desc  # bytes array
        greedy = self.Q.argmax(axis=1)

        # deterministic delta mapping for actions: 0=LEFT,1=DOWN,2=RIGHT,3=UP
        deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        # Rollout from Start (state 0) deterministically using greedy actions (no slip)
        s = 0
        path = [s]
        for _ in range(max_steps):
            r, c = divmod(s, ncol)
            tile = desc[r][c].decode()
            if tile in ('H', 'G'):
                # include terminal tile then stop
                break
            a = greedy[s]
            dr, dc = deltas[a]
            nr = min(max(r + dr, 0), nrow - 1)
            nc = min(max(c + dc, 0), ncol - 1)
            ns = nr * ncol + nc
            path.append(ns)
            s = ns
            rr, cc = divmod(s, ncol)
            if desc[rr][cc].decode() in ('H', 'G'):
                break

        visited = set(path)

        # Build base grid letters (ASCII-friendly)
        base_grid = np.full((nrow, ncol), '-', dtype='<U4')
        for i in range(nrow):
            for j in range(ncol):
                ch = desc[i][j].decode()
                if ch in ('S', 'H', 'G'):
                    base_grid[i, j] = ch
                else:
                    base_grid[i, j] = 'F'  # frozen tile

        # ASCII fallback: print with G(o)/H(o) when terminal visited
        if ax is None:
            viz = base_grid.copy()
            for idx in visited:
                i, j = divmod(idx, ncol)
                ch = desc[i][j].decode()
                if ch == 'G':
                    viz[i, j] = 'G(o)'
                elif ch == 'H':
                    viz[i, j] = 'H(o)'
                elif ch == 'S':
                    viz[i, j] = 'S'
                else:
                    viz[i, j] = 'o'
            viz[0, 0] = 'S'
            print("\nGreedy Path Grid (ASCII):")
            for row in viz:
                print(" ".join(f"{cell:3}" for cell in row))
            return

        # --- Matplotlib rendering: color visited cells, then draw tile letters on top ---
        ax.imshow(np.zeros((nrow, ncol)), cmap=plt.get_cmap("Greys"), vmin=0, vmax=1)

        # Fill visited cells first (so letters appear above)
        for idx in visited:
            i, j = divmod(idx, ncol)
            ch = desc[i][j].decode()
            if ch == 'G':
                face = goal_color
            elif ch == 'H':
                face = hole_color
            elif ch == 'S':
                face = "#c8f7c5"  # subtle green for start if visited
            else:
                face = highlight_color
            rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                    facecolor=face, edgecolor=None, alpha=0.9, zorder=0)
            ax.add_patch(rect)

        # Draw grid lines
        ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
        ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=1, alpha=0.5)

        # Draw letters on top (base tiles) with clear colors
        for i in range(nrow):
            for j in range(ncol):
                ch = desc[i][j].decode()
                color = "black"
                if ch == 'S':
                    color = "darkgreen"
                elif ch == 'G':
                    color = "darkgoldenrod"
                elif ch == 'H':
                    color = "darkred"
                ax.text(j, i, ch, ha='center', va='center', fontsize=16, fontweight='bold', color=color, zorder=2)

        # --- Highlight final greedy state with a rectangle border UNDER the letters (zorder=1) ---
        final_idx = path[-1]
        fi, fj = divmod(final_idx, ncol)
        final_tile = desc[fi][fj].decode()
        if final_tile == 'G':
            edgecol = "darkgoldenrod"
        elif final_tile == 'H':
            edgecol = "darkred"
        else:
            edgecol = "k"
        # border rectangle, no fill, zorder=1 so letters (zorder=2) remain on top
        highlight_rect = mpatches.Rectangle((fj - 0.5, fi - 0.5), 1.0, 1.0,
                                            fill=False, edgecolor=edgecol, linewidth=2.0, zorder=1)
        ax.add_patch(highlight_rect)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, ncol - 0.5)
        ax.set_ylim(nrow - 0.5, -0.5)  # flip so row 0 is top
        ax.set_title("Greedy Path Grid")


    def print_greedy_path_grid(self, ax=None, max_steps: int = 200,
                                   visited_color: str = "#cfefff",
                                   start_color: str = "#c8f7c5",
                                   goal_color: str = "#ffd86b",
                                   hole_color: str = "#ffb3b3"):
        """
        Very simple deterministic greedy rollout using the SAME mapping as plot_q_heatmap:
        next_state = clamp(r+dr, c+dc) for action a.
        Stops on terminal ('H'/'G'), cycle detection, or max_steps.
        Colors the visited cells in path order.
        Returns the path (list of states).
        """
        import matplotlib.patches as mpatches

        env = self.env.unwrapped
        nrow, ncol = env.nrow, env.ncol
        desc = env.desc  # bytes
        # Reuse the unified greedy path computation
        path = self.compute_greedy_path(max_steps=max_steps)
        print("Greedy path (states):", path)

        # If no axis given, just return path (or print ASCII)
        if ax is None:
            print("Greedy path (states):", path)
            return path

        # --- Simple plotting (color path in order) ---
        ax.imshow(np.zeros((nrow, ncol)), cmap=plt.get_cmap("Greys"), vmin=0, vmax=1)

        # color visited in path order (later states darker)
        cmap = plt.get_cmap("Blues")
        total = max(1, len(path))
        for idx_pos, st in enumerate(path):
            i, j = divmod(st, ncol)
            tile = desc[i][j].decode()
            if tile == "S":
                face = start_color
            elif tile == "G":
                face = goal_color
            elif tile == "H":
                face = hole_color
            else:
                # use a shade from the colormap to show order
                shade = cmap(0.25 + 0.65 * (idx_pos / (total - 1))) if total > 1 else cmap(0.6)
                face = shade
            rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=face, edgecolor='k', zorder=0)
            ax.add_patch(rect)

        # draw tile letters on top
        for i in range(nrow):
            for j in range(ncol):
                ch = desc[i][j].decode()
                color = "black"
                if ch == 'S': color = "darkgreen"
                if ch == 'G': color = "darkgoldenrod"
                if ch == 'H': color = "darkred"
                ax.text(j, i, ch, ha='center', va='center', fontsize=12, fontweight='bold', color=color, zorder=2)

        # highlight final state
        final = path[-1]
        fi, fj = divmod(final, ncol)
        final_tile = desc[fi][fj].decode()
        edgecol = "darkgoldenrod" if final_tile == 'G' else ("darkred" if final_tile == 'H' else 'k')
        highlight = mpatches.Rectangle((fj - 0.5, fi - 0.5), 1.0, 1.0, fill=False, edgecolor=edgecol, linewidth=2.5, zorder=3)
        ax.add_patch(highlight)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, ncol - 0.5)
        ax.set_ylim(nrow - 0.5, -0.5)  # flip so row 0 is top
        ax.set_title("Greedy Path (clamped mapping, deterministic)")

        return path


    def _action_name(self, action):
        # Helper to map action index to name
        names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        return names.get(action, str(action))

    def compute_greedy_path(self, max_steps: int = 200) -> list[int]:
        env = self.env.unwrapped
        nrow, ncol = env.nrow, env.ncol
        desc = env.desc
        greedy_actions = np.argmax(self.Q, axis=1)
        deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        s = 0
        path: list[int] = [s]
        seen_states: set[int] = {s}
        for _ in range(max_steps):
            r, c = divmod(s, ncol)
            tile = desc[r][c].decode()
            if tile in ("H", "G"):
                break
            a = int(greedy_actions[s])
            dr, dc = deltas[a]
            nr, nc = r + dr, c + dc
            nr = min(max(nr, 0), nrow - 1)
            nc = min(max(nc, 0), ncol - 1)
            ns = nr * ncol + nc
            path.append(ns)
            s = ns
            if s in seen_states:
                break
            seen_states.add(s)
            r, c = divmod(s, ncol)
            if desc[r][c].decode() in ("H", "G"):
                break
        return path

    def evaluate_policy_metrics(self, num_episodes: int = 100, gamma: float = 0.99, pi: nn.Module = None) -> dict:
        """
        Evaluate the current greedy policy (derived from self.Q) over num_episodes.
        Returns a dict with success_rate, avg_gamma_return, and avg_trajectory_length.
        """
        successes = 0
        returns = []
        lengths = []
        for _ in range(num_episodes):
            if pi is None:
                traj, _ = self.sample_trajectory(epsilon_greedy=False)
            else:
                traj = self.sample_trajectory_pi_nn(pi, max_steps=200)
            if len(traj) > 0 and traj[-1][2] == self.goal_reward:
                successes += 1
            G = sum(r * (gamma ** t) for t, (_, _, r) in enumerate(traj))
            returns.append(G)
            lengths.append(len(traj))
        return {
            "success_rate": successes / num_episodes if num_episodes > 0 else 0.0,
            "avg_gamma_return": float(np.mean(returns)) if len(returns) > 0 else 0.0,
            "avg_trajectory_length": float(np.mean(lengths)) if len(lengths) > 0 else 0.0,
        }

    def success_rate(self, num_episodes: int = 100) -> float:
        """
        Backward-compat helper: returns only success rate of current greedy policy.
        """
        metrics = self.evaluate_policy_metrics(num_episodes=num_episodes, gamma=0.99)
        return metrics["success_rate"]

    def plot_q_heatmap(
        self,
        Q: np.ndarray | None = None,
        title: str = "Q-Value Heatmap",
        show: bool = True,
        block: bool = True,
        annot_fontsize: int = 7,
        xtick_fontsize: int = 11,
        ytick_fontsize: int = 8,
        figsize: tuple = (14, 12),
    ):
        """
        Plots a heatmap of Q-values (state-action values), and annotates each cell with
        (Q value, next state) where next state is the state the agent would land in if it
        takes that action from the current state, based on the grid layout (not transition probabilities).

        Args:
            Q: Optional array of shape (n_states, n_actions). If None, uses self.Q.
            title: Title for the plot.
            show: If True, calls plt.show().
            block: Passed to plt.show(block=...). Ignored if show is False.
            annot_fontsize: Font size for cell annotations.
            xtick_fontsize: Font size for x-axis (action) labels.
            ytick_fontsize: Font size for y-axis (state) labels.
            figsize: Figure size.
        """
        if Q is None:
            Q = self.Q

        n_states, n_actions = Q.shape
        nrow = self.actual_env.nrow
        ncol = self.actual_env.ncol

        # Action deltas: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
        deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        annot = np.empty(Q.shape, dtype=object)
        for s in range(n_states):
            r, c = divmod(s, ncol)
            for a in range(n_actions):
                dr, dc = deltas[a]
                nr, nc = r + dr, c + dc
                # Clamp to grid boundaries
                nr = min(max(nr, 0), nrow - 1)
                nc = min(max(nc, 0), ncol - 1)
                next_state = nr * ncol + nc
                # Format: Q-value (next_state)
                annot[s, a] = f"{Q[s, a]:.2f} ({next_state})"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            Q,
            annot=annot,
            fmt='',
            cmap="Blues",
            cbar=True,
            xticklabels=["←", "↓", "→", "↑"],
            yticklabels=[f"S{i}" for i in range(Q.shape[0])],
            annot_kws={"fontsize": annot_fontsize},
            ax=ax,
            linewidths=1,
            linecolor='black'
        )
        ax.set_title(title, fontsize=15, pad=16)
        ax.set_xlabel("Actions", fontsize=xtick_fontsize + 2)
        ax.set_ylabel("States", fontsize=ytick_fontsize + 2)
        ax.tick_params(axis='x', labelsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)
        plt.tight_layout()
        if show:
            try:
                plt.show(block=block)
            except TypeError:
                plt.show()
        return fig, ax

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

        # Success rate (left) and eval-based avg traj length (right)
        if len(sr_indices) > 0:
            ax_sr.plot(sr_indices, sr_values, marker='o', color='blue', label='Success Rate (checkpoint)')
            ax_sr.set_ylim(-0.05, 1.05)
        else:
            ax_sr.plot([], [], label='No Success Rate Data')

        # Secondary y-axis: evaluation avg trajectory length at checkpoints
        eval_lengths = np.array(logger.get("eval_avg_traj_lengths", []))
        if eval_lengths.size > 0:
            ax_sr_r = ax_sr.twinx()
            # plot only non -1 entries aligned on x where checkpoints were logged
            xs = [i for i, v in enumerate(eval_lengths) if v != -1.0]
            ys = [eval_lengths[i] for i in xs]
            if len(xs) > 0:
                ax_sr_r.plot(xs, ys, color='gray', marker='s', alpha=0.7, label='Avg Traj Length (eval)')
                ax_sr_r.set_ylabel('Avg Trajectory Length (eval)')
                ax_sr_r.set_ylim(0, 100)
                # merge legends
                h1, l1 = ax_sr.get_legend_handles_labels()
                h2, l2 = ax_sr_r.get_legend_handles_labels()
                ax_sr.legend(h1 + h2, l1 + l2, loc='upper right')
            else:
                ax_sr.legend()
        else:
            ax_sr.legend()
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
        ax_metrics = fig.add_subplot(gs[2, 1])
        ax_metrics.axis('off')
        ax_metrics.set_title("Summary Metrics")
        fm = logger.get("final_metrics", {}) or {}
        lines = []
        def fmt(value):
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value) if value is not None else "N/A"
        lines.append(f"Best policy iteration: {fmt(fm.get('best_policy_iteration'))}")
        lines.append(f"Best policy success rate: {fmt(fm.get('best_policy_success_rate'))}")
        lines.append(f"Best policy avg gamma return: {fmt(fm.get('best_policy_avg_gamma_return'))}")
        lines.append("")
        lines.append(f"Final policy success rate: {fmt(fm.get('final_policy_success_rate'))}")
        lines.append(f"Final policy avg gamma return: {fmt(fm.get('final_policy_avg_gamma_return'))}")
        avg_len = fm.get('avg_trajectory_length', fm.get('final_policy_avg_trajectory_length'))
        if avg_len is not None:
            lines.append(f"Final policy avg trajectory length: {fmt(avg_len)}")
        try:
            greedy_path_states = self.compute_greedy_path(max_steps=200)
            # Break path into chunks of 6 states per line
            for i in range(0, len(greedy_path_states), 6):
                chunk = greedy_path_states[i:i+6]
                if i == 0:
                    lines.append(f"Greedy path (states):")
                lines.append(f"    {chunk}")
        except Exception:
            pass
        if len(lines) == 0:
            ax_metrics.text(0.5, 0.5, "No summary metrics found", ha='center', va='center', fontsize=12)
        else:
            ax_metrics.text(0.0, 1.0, "\n".join(lines), ha='left', va='top', fontsize=12, family='monospace')

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