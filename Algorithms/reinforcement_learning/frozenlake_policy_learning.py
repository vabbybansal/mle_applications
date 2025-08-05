import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FrozenLakeEnv(gym.Env):
    def __init__(self, 
                 map_name="8x8", 
                 is_slippery=True,
                 step_penalty=-0.01,
                 hole_penalty=-0.5,
                 goal_reward=1.0,
                 gamma=0.9,
                 theta=1e-8
        ):
        self.gamma = gamma
        self.theta = theta
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

        # print(self.actual_env.P)

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # self.policy = np.zeros(self.n_states, dtype=int) # policy is a vector of length n_states, where each element is an action - Deterministic policy
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions # Stochastic policy

        self.value_function = np.zeros(self.n_states) # value function is a vector of length n_states, where each element is a scalar

        # self.print_transition_probabilities(self.actual_env)

        # self.env.P = 
        self.set_slip_probabilities(0.8, 0.1, 0.1)
        self.modify_rewards(self.actual_env)    # modify the rewards of the environment to include the step penalty, hole penalty, and goal reward

        self.logs = {
            "value_deltas": [],
            "policy_changes": [],
            "eval_num_iters": [],
            "policy": [],
            "value_function": []
        }

        # self.print_transition_probabilities(self.actual_env)

    
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
                    (p_intended, ns_i, reward(ns_i), False),
                    (p_left,     ns_l, reward(ns_l), False),
                    (p_right,    ns_r, reward(ns_r), False),
                ]

    def policy_evaluation(self):
        eval_num_iters = 0
        while True:
            delta = 0
            # make a fresh copy to hold V_{k+1}
            new_V = np.zeros_like(self.value_function)

            for state, action_dict in self.actual_env.P.items():
                v = 0
                for action, transitions in action_dict.items():
                    for prob, next_state, reward, _ in transitions:
                        v += self.policy[state, action] * prob * (
                                reward + self.gamma * self.value_function[next_state]
                            )
                new_V[state] = v
                delta = max(delta, abs(v - self.value_function[state]))
                self.logs["value_deltas"].append(delta)

            self.value_function = new_V
            eval_num_iters += 1
            if delta < self.theta:
                self.logs["eval_num_iters"].append(eval_num_iters)
                break
                
    def policy_improvement_greedy(self):
        policy_stable = True

        changes = 0

        new_policy = np.zeros((self.n_states, self.n_actions))
        # for all states, find the best action by looking at the adjacent state value functions
        for state in self.actual_env.P:
            old_action = np.argmax(self.policy[state])
            action_values = np.zeros(self.n_actions)
            for action in self.actual_env.P[state]:
                max_v_next = -np.inf
                best_action_next = None
                # for all actions, find the best action by looking at the adjacent state value functions
                for prob_ssa, next_state, reward_sas, _ in self.actual_env.P[state][action]:     # prob_ss'a, next_state, reward_sas', is_done
                    # even if we choose action a, it can lead to multiple next states with different prob and rewards and values.. hence we find the expected action value across states
                    action_values[action] += prob_ssa * (reward_sas + self.gamma * self.value_function[next_state])
                    # if the value function for this next state is better, then update the best action as the one that leads to this next state
                    # update the new best value
            best_action_next = np.argmax(action_values)
            if old_action != best_action_next:
                changes += 1

            # greedy policy update: prob of the best action becomes 1 and all other action prob become 0
            new_policy[state][best_action_next] = 1 

        # check if the policy is stable or not
        if not np.array_equal(self.policy, new_policy):
            policy_stable = False
        self.policy = new_policy
        self.logs["policy_changes"].append(changes)
        return policy_stable
    

    def policy_improvement_stochastic(self, tau=0.1):
        policy_stable = True
        changes = 0
        new_policy = np.zeros((self.n_states, self.n_actions))

        for state in range(self.n_states):
            old_action = np.argmax(self.policy[state])
            action_values = np.zeros(self.n_actions)

            for action in range(self.n_actions):
                for prob, next_state, reward, _ in self.actual_env.P[state][action]:
                    action_values[action] += prob * (reward + self.gamma * self.value_function[next_state])

            # Softmax policy over action-values
            exp_vals = np.exp(action_values / tau)
            new_policy[state] = exp_vals / np.sum(exp_vals)

            new_action = np.argmax(new_policy[state])
            if old_action != new_action:
                changes += 1
                policy_stable = False

        self.policy = new_policy
        self.logs["policy_changes"].append(changes)
        return policy_stable

    
    def policy_iteration(self, use_greedy=True):
        
        self.logs["policy"].append(self.policy.copy())
        self.logs["value_function"].append(self.value_function.copy())

        idx = 0
        while True:
            print(f"Policy Evaluation step - {idx}")
            self.policy_evaluation()

            print(f"Policy Improvement step - {idx}")
            if use_greedy:
                policy_stable = self.policy_improvement_greedy()
            else:
                policy_stable = self.policy_improvement_stochastic()
            if policy_stable:
                break
            self.logs["policy"].append(self.policy.copy())
            self.logs["value_function"].append(self.value_function.copy())
            idx += 1
        
        # Draw plots
        lake_obj.plot_value_and_policy(lake_obj.value_function, lake_obj.policy, lake_obj.actual_env.desc)
        lake_obj.plot_convergence()
        lake_obj.plot_policy_heatmap(lake_obj.policy)

    def plot_value_and_policy(self, value_function, policy, env_desc, title="Value Function + Greedy Policy Plan", ax=None):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import colors

        nrow, ncol = env_desc.shape
        reshaped_values = value_function.reshape(nrow, ncol)
        policy_grid = np.array([np.argmax(policy[s]) for s in range(value_function.shape[0])]).reshape(nrow, ncol)

        action_map = ['←', '↓', '→', '↑']

        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            show_plot = True

        im = ax.imshow(reshaped_values, cmap="coolwarm", origin="upper")

        for r in range(nrow):
            for c in range(ncol):
                s = r * ncol + c
                tile = env_desc[r][c].decode("utf-8")
                color = 'black'
                if tile == 'H':
                    ax.text(c, r, f"H\n{value_function[s]:.2f}", ha='center', va='center',
                            color='red', fontsize=10, fontweight='bold')
                elif tile == 'G':
                    ax.text(c, r, f"G\n{value_function[s]:.2f}", ha='center', va='center',
                            color='green', fontsize=10, fontweight='bold')
                elif tile == 'S':
                    action = policy_grid[r, c]
                    ax.text(c, r, f"S {action_map[action]}\n{value_function[s]:.2f}", ha='center', va='center',
                            color='blue', fontsize=10, fontweight='bold')
                else:
                    action = policy_grid[r, c]
                    ax.text(c, r, f"{action_map[action]}\n{value_function[s]:.2f}", ha='center', va='center',
                            color=color, fontsize=10)

        # Trace and plot the best path
        start_state = np.where(env_desc.flatten() == b'S')[0][0]
        current_state = start_state
        visited = set()
        path = [current_state]

        while True:
            if current_state in visited:
                break  # avoid loops
            visited.add(current_state)

            r, c = divmod(current_state, ncol)
            tile = env_desc[r][c].decode('utf-8')
            if tile in ['G', 'H']:
                break

            action = np.argmax(policy[current_state])
            if action == 0:  # LEFT
                next_state = current_state - 1
            elif action == 1:  # DOWN
                next_state = current_state + ncol
            elif action == 2:  # RIGHT
                next_state = current_state + 1
            elif action == 3:  # UP
                next_state = current_state - ncol
            else:
                break

            if next_state < 0 or next_state >= value_function.shape[0]:
                break

            path.append(next_state)
            current_state = next_state

        for i in range(len(path) - 1):
            r1, c1 = divmod(path[i], ncol)
            r2, c2 = divmod(path[i+1], ncol)
            ax.arrow(c1, r1, c2 - c1, r2 - r1, head_width=0.2, head_length=0.2, fc='orange', ec='orange')

        ax.set_xticks(np.arange(ncol))
        ax.set_yticks(np.arange(nrow))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.grid(False)
        if show_plot:
            plt.show()

    def plot_convergence(self):
        value_deltas = self.logs["value_deltas"]
        policy_changes = self.logs["policy_changes"]
        eval_num_iters = self.logs["eval_num_iters"]

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))

        axs[0].plot(value_deltas, marker='o')
        axs[0].set_title("Value Function Convergence")
        axs[0].set_xlabel("Evaluation Iteration")
        axs[0].set_ylabel("Max Δ")

        axs[1].plot(policy_changes, marker='s', color='green')
        axs[1].set_title("Policy Changes per Iteration")
        axs[1].set_xlabel("Policy Improvement Iteration")
        axs[1].set_ylabel("Policy Changes")

        axs[2].plot(eval_num_iters, marker='s', color='blue')
        axs[2].set_title("Policy Evaluation Num Iterations")
        axs[2].set_xlabel("Policy Iteration Outer Loop Iteration Number")
        axs[2].set_ylabel("Policy Evaluation Inner Loop Numeber of Iterations")

        # Plot value and policy for the last up to 4 iterations if stored in logs
        if "value_function" in self.logs and "policy" in self.logs:
            n_iters = min(len(self.logs["value_function"]), len(self.logs["policy"]))
            if n_iters > 0:
                last_n = min(4, n_iters)
                value_functions = self.logs["value_function"][-last_n:]
                policies = self.logs["policy"][-last_n:]
                fig, axes = plt.subplots(1, last_n, figsize=(4 * last_n, 4))
                if last_n == 1:
                    axes = np.array([axes])
                for i in range(last_n):
                    value_fn = value_functions[i]
                    policy = policies[i]
                    ax = axes[i]
                    # Call plot_value_and_policy with ax as a keyword argument for ax to avoid TypeError
                    self.plot_value_and_policy(
                        value_fn,
                        policy,
                        self.actual_env.desc,
                        f"Value & Policy at Iteration {n_iters - last_n + i + 1}",
                        ax=ax
                    )
                plt.tight_layout()
                plt.show()

        plt.show()

    def plot_policy_only(self, title="Policy Only"):

        print(self.policy)
        det_policy = np.argmax(self.policy, axis=1).reshape((self.actual_env.nrow, self.actual_env.ncol))
        print(det_policy)


    def plot_policy_heatmap(self, policy, title="Policy Heatmap"):
        """
        Plots a heatmap of the policy action probabilities for each state.

        Args:
            policy: numpy array of shape (n_states, n_actions) with action probabilities
            title: Title for the plot
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            policy,
            annot=True,
            cmap="Blues",
            cbar=True,
            xticklabels=["←", "↓", "→", "↑"],
            yticklabels=[f"S{i}" for i in range(policy.shape[0])]
        )
        plt.title(title)
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.tight_layout()
        plt.show()


lake_obj = FrozenLakeEnv()
print(lake_obj)
lake_obj.policy_iteration(use_greedy=True)
lake_obj.policy_iteration(use_greedy=False)