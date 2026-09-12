import numpy as np

# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
#  .  free aisle        #  racking (impassable)
#  S  loading bay       G  despatch station
#  C  conveyor strip (heavy penalty, ends the episode)
LAYOUT = [
    "............",
    ".####..####.",
    ".####..####.",
    "............",
    ".####..####.",
    ".####..####.",
    "............",
    "SCCCCCCCCCCG",
]

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

STEP_REWARD = -1.0
CONVEYOR_REWARD = -100.0
GOAL_REWARD = 0.0
SLIP_PROB = 0.0          # set above 0 for a stochastic floor


class WarehouseGrid:
    def __init__(self, layout=LAYOUT, slip_prob=SLIP_PROB, rng=None):
        self.grid = [list(row) for row in layout]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.slip_prob = slip_prob
        self.rng = rng or np.random.default_rng()

        self.start = self._find("S")
        self.goal = self._find("G")
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(ACTIONS)

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            for c, v in enumerate(row):
                if v == ch:
                    return (r, c)
        raise ValueError(f"cell {ch!r} not present in the layout")

    def to_index(self, pos):
        return pos[0] * self.n_cols + pos[1]

    def to_pos(self, idx):
        return divmod(idx, self.n_cols)

    def is_wall(self, r, c):
        return not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or self.grid[r][c] == "#"

    def reset(self):
        self.pos = self.start
        return self.to_index(self.pos)

    def step(self, action_idx):
        action = ACTIONS[action_idx]

        # a slippery floor occasionally rotates the intended move
        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            action = ACTIONS[int(self.rng.integers(self.n_actions))]

        dr, dc = DELTAS[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc
        if self.is_wall(nr, nc):
            nr, nc = r, c               # bump into racking, stay put

        cell = self.grid[nr][nc]
        if cell == "C":
            # swept onto the conveyor: penalty and the episode ends back at start
            self.pos = self.start
            return self.to_index(self.start), CONVEYOR_REWARD, True
        if cell == "G":
            self.pos = (nr, nc)
            return self.to_index(self.pos), GOAL_REWARD, True

        self.pos = (nr, nc)
        return self.to_index(self.pos), STEP_REWARD, False


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
def epsilon_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


# --------------------------------------------------------------------------
# TD(0) prediction
# --------------------------------------------------------------------------
def td0_prediction(env, policy, episodes=2000, alpha=0.1, gamma=0.95,
                   max_steps=200, rng=None):
    """Estimate V(s) for the supplied policy. `policy` maps state index -> action."""
    rng = rng or np.random.default_rng()
    V = np.zeros(env.n_states)

    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = policy(s, rng)
            s_next, r, done = env.step(a)
            target = r if done else r + gamma * V[s_next]
            V[s] += alpha * (target - V[s])
            if done:
                break
            s = s_next

    return V


# --------------------------------------------------------------------------
# SARSA — on-policy control
# --------------------------------------------------------------------------
def sarsa(env, episodes=3000, alpha=0.1, gamma=0.95, epsilon=0.1,
          max_steps=200, rng=None):
    rng = rng or np.random.default_rng()
    Q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(episodes)

    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q[s], epsilon, rng)
        total = 0.0

        for _ in range(max_steps):
            s_next, r, done = env.step(a)
            total += r
            if done:
                Q[s, a] += alpha * (r - Q[s, a])
                break
            # the next action is chosen by the same behaviour policy, and that
            # choice is what appears in the update target
            a_next = epsilon_greedy(Q[s_next], epsilon, rng)
            Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
            s, a = s_next, a_next

        returns[ep] = total

    return Q, returns


# --------------------------------------------------------------------------
# Q-Learning — off-policy control
# --------------------------------------------------------------------------
def q_learning(env, episodes=3000, alpha=0.1, gamma=0.95, epsilon=0.1,
               max_steps=200, rng=None):
    rng = rng or np.random.default_rng()
    Q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(episodes)

    for ep in range(episodes):
        s = env.reset()
        total = 0.0

        for _ in range(max_steps):
            a = epsilon_greedy(Q[s], epsilon, rng)
            s_next, r, done = env.step(a)
            total += r
            if done:
                Q[s, a] += alpha * (r - Q[s, a])
                break
            # the target uses the greedy action, not the one actually taken
            Q[s, a] += alpha * (r + gamma * Q[s_next].max() - Q[s, a])
            s = s_next

        returns[ep] = total

    return Q, returns


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
ARROW = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}


def render_policy(env, Q):
    out = []
    for r in range(env.n_rows):
        row = []
        for c in range(env.n_cols):
            cell = env.grid[r][c]
            if cell == "#":
                row.append("#")
            elif cell == "C":
                row.append("C")
            elif cell == "G":
                row.append("G")
            else:
                a = int(Q[env.to_index((r, c))].argmax())
                row.append(ARROW[ACTIONS[a]])
        out.append(" ".join(row))
    return "\n".join(out)


def render_values(env, V):
    out = []
    for r in range(env.n_rows):
        row = []
        for c in range(env.n_cols):
            if env.grid[r][c] == "#":
                row.append("      #")
            else:
                row.append(f"{V[env.to_index((r, c))]:7.1f}")
        out.append(" ".join(row))
    return "\n".join(out)


def greedy_rollout(env, Q, max_steps=200):
    """Run the greedy policy once and return (path, total reward, reached_goal)."""
    s = env.reset()
    path = [env.to_pos(s)]
    total = 0.0
    for _ in range(max_steps):
        a = int(Q[s].argmax())
        s, r, done = env.step(a)
        total += r
        path.append(env.to_pos(s))
        if done:
            return path, total, env.grid[path[-1][0]][path[-1][1]] == "G"
    return path, total, False


def main():
    rng = np.random.default_rng(42)
    env = WarehouseGrid(rng=rng)

    print("Warehouse robot navigation — TD(0), SARSA and Q-Learning")
    print("=" * 64)
    print("Floor plan   S start   G goal   # racking   C conveyor hazard\n")
    print("\n".join(" ".join(row) for row in env.grid))

    episodes = 3000
    alpha, gamma, epsilon = 0.1, 0.95, 0.1
    print(f"\nepisodes={episodes}  alpha={alpha}  gamma={gamma}  epsilon={epsilon}")

    # ---- TD(0) prediction under a uniformly random policy -----------------
    print("\n" + "-" * 64)
    print("TD(0) prediction — V(s) for a uniformly random policy")
    print("-" * 64)
    random_policy = lambda s, r: int(r.integers(env.n_actions))
    V_random = td0_prediction(env, random_policy, episodes=episodes,
                              alpha=alpha, gamma=gamma, rng=np.random.default_rng(1))
    print(render_values(env, V_random))
    print("\nValues are strongly negative near the conveyor: a random walk")
    print("falls onto it often, and TD(0) reports that faithfully.")

    # ---- SARSA ------------------------------------------------------------
    print("\n" + "-" * 64)
    print("SARSA — on-policy control")
    print("-" * 64)
    Q_sarsa, ret_sarsa = sarsa(env, episodes=episodes, alpha=alpha, gamma=gamma,
                               epsilon=epsilon, rng=np.random.default_rng(2))
    print(render_policy(env, Q_sarsa))

    # ---- Q-Learning -------------------------------------------------------
    print("\n" + "-" * 64)
    print("Q-Learning — off-policy control")
    print("-" * 64)
    Q_ql, ret_ql = q_learning(env, episodes=episodes, alpha=alpha, gamma=gamma,
                              epsilon=epsilon, rng=np.random.default_rng(3))
    print(render_policy(env, Q_ql))

    # ---- comparison -------------------------------------------------------
    tail = episodes // 10
    print("\n" + "-" * 64)
    print("Comparison")
    print("-" * 64)
    print(f"{'algorithm':<14}{'mean return (last 10%)':>26}{'greedy path':>14}{'return':>10}")

    for name, Q, ret in (("SARSA", Q_sarsa, ret_sarsa), ("Q-Learning", Q_ql, ret_ql)):
        path, total, reached = greedy_rollout(env, Q)
        status = f"{len(path) - 1} steps" if reached else "no goal"
        print(f"{name:<14}{ret[-tail:].mean():>26.2f}{status:>14}{total:>10.1f}")

    print("\nDuring training SARSA usually shows the better average return,")
    print("because it accounts for the exploratory steps that push the robot")
    print("onto the conveyor. Q-Learning recovers the shorter greedy route")
    print("along the conveyor edge, which is optimal only once exploration stops.")

    print("\nGreedy path found by Q-Learning:")
    path, _, _ = greedy_rollout(env, Q_ql)
    print("  " + " -> ".join(f"({r},{c})" for r, c in path))


if __name__ == "__main__":
    main()
