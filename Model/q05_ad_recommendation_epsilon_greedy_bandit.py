import numpy as np
AD_NAMES = [
    "A: banner, free-trial offer",
    "B: banner, discount code",
    "C: video, product demo",
    "D: carousel, testimonials",
    "E: static, brand awareness",
]

TRUE_CTR = np.array([0.021, 0.035, 0.052, 0.028, 0.011])

HORIZON = 10_000        # impressions per campaign
N_RUNS = 100            # independent campaigns, averaged over
SEED = 7


# --------------------------------------------------------------------------
# the agent
# --------------------------------------------------------------------------
class EpsilonGreedyBandit:
    """Incremental-average epsilon-greedy bandit.

    epsilon      fixed exploration rate, or None when a schedule is supplied
    schedule     callable t -> epsilon, for decaying exploration
    initial      initial value of every arm's estimate; a value above the
                 highest possible reward makes the agent optimistic and so
                 explores every arm before settling
    """

    def __init__(self, n_arms, epsilon=0.1, schedule=None, initial=0.0, rng=None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.schedule = schedule
        self.rng = rng or np.random.default_rng()
        self.q = np.full(n_arms, float(initial))   # value estimates
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.t = 0

    def current_epsilon(self):
        if self.schedule is not None:
            return self.schedule(self.t)
        return self.epsilon

    def select(self):
        self.t += 1
        eps = self.current_epsilon()
        if eps > 0.0 and self.rng.random() < eps:
            return int(self.rng.integers(self.n_arms))
        # ties broken at random so the first arm is not favoured
        best = np.flatnonzero(self.q == self.q.max())
        return int(self.rng.choice(best))

    def update(self, arm, reward):
        self.counts[arm] += 1
        # incremental mean: Q <- Q + (r - Q) / n
        self.q[arm] += (reward - self.q[arm]) / self.counts[arm]


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def run_campaign(agent_factory, true_ctr, horizon, rng):
    """Run one campaign and return (cumulative reward, cumulative regret,
    optimal-arm share, final estimates)."""
    agent = agent_factory(rng)
    best_ctr = true_ctr.max()
    best_arm = int(true_ctr.argmax())

    total_reward = 0.0
    total_regret = 0.0
    optimal_pulls = 0

    for _ in range(horizon):
        arm = agent.select()
        reward = 1.0 if rng.random() < true_ctr[arm] else 0.0
        agent.update(arm, reward)

        total_reward += reward
        total_regret += best_ctr - true_ctr[arm]
        optimal_pulls += (arm == best_arm)

    return total_reward, total_regret, optimal_pulls / horizon, agent.q.copy()


def evaluate(name, agent_factory, true_ctr, horizon, n_runs, seed):
    rewards, regrets, shares, estimates = [], [], [], []
    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        r, g, s, q = run_campaign(agent_factory, true_ctr, horizon, rng)
        rewards.append(r)
        regrets.append(g)
        shares.append(s)
        estimates.append(q)

    return {
        "name": name,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "regret_mean": float(np.mean(regrets)),
        "regret_std": float(np.std(regrets)),
        "optimal_share": float(np.mean(shares)),
        "estimates": np.mean(estimates, axis=0),
    }


def main():
    n_arms = len(TRUE_CTR)
    best_arm = int(TRUE_CTR.argmax())

    print("Epsilon-greedy Multi-Armed Bandit — online ad recommendation")
    print("=" * 72)
    print(f"Creatives : {n_arms}")
    print(f"Horizon   : {HORIZON:,} impressions per campaign")
    print(f"Campaigns : {N_RUNS} independent runs, results averaged")
    print("\nTrue click-through rates (unknown to the agent)")
    for i, (nm, p) in enumerate(zip(AD_NAMES, TRUE_CTR)):
        mark = "  <- optimal" if i == best_arm else ""
        print(f"  {nm:<32} {p:.4f}{mark}")

    configs = [
        ("greedy (eps = 0.00)",
         lambda rng: EpsilonGreedyBandit(n_arms, epsilon=0.00, rng=rng)),
        ("eps-greedy (eps = 0.01)",
         lambda rng: EpsilonGreedyBandit(n_arms, epsilon=0.01, rng=rng)),
        ("eps-greedy (eps = 0.05)",
         lambda rng: EpsilonGreedyBandit(n_arms, epsilon=0.05, rng=rng)),
        ("eps-greedy (eps = 0.10)",
         lambda rng: EpsilonGreedyBandit(n_arms, epsilon=0.10, rng=rng)),
        ("eps-greedy (eps = 0.30)",
         lambda rng: EpsilonGreedyBandit(n_arms, epsilon=0.30, rng=rng)),
        ("decaying eps = 1/(1 + t/500)",
         lambda rng: EpsilonGreedyBandit(
             n_arms, epsilon=None, schedule=lambda t: 1.0 / (1.0 + t / 500.0), rng=rng)),
        ("optimistic init (Q0 = 0.10, greedy)",
         lambda rng: EpsilonGreedyBandit(n_arms, epsilon=0.00, initial=0.10, rng=rng)),
    ]

    results = [evaluate(nm, fac, TRUE_CTR, HORIZON, N_RUNS, SEED) for nm, fac in configs]

    print("\nResults")
    print("-" * 72)
    print(f"{'strategy':<36}{'clicks':>12}{'regret':>12}{'opt. share':>12}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: -x["reward_mean"]):
        print(f"{r['name']:<36}"
              f"{r['reward_mean']:>9.1f}±{r['reward_std']:<4.0f}"
              f"{r['regret_mean']:>9.1f}±{r['regret_std']:<4.0f}"
              f"{r['optimal_share'] * 100:>11.1f}%")

    oracle = TRUE_CTR.max() * HORIZON
    print("-" * 72)
    print(f"{'oracle (always show the best ad)':<36}{oracle:>9.1f}{0.0:>16.1f}{100.0:>11.1f}%")

    winner = max(results, key=lambda x: x["reward_mean"])
    print(f"\nBest strategy: {winner['name']}")
    print("Final CTR estimates under that strategy")
    for nm, est, true in zip(AD_NAMES, winner["estimates"], TRUE_CTR):
        print(f"  {nm:<32} estimate {est:.4f}   true {true:.4f}")

    print("\nReading the table: pure greedy locks onto whichever creative")
    print("happened to click first and never recovers, while a large epsilon")
    print("keeps paying for exploration long after the answer is known. The")
    print("useful settings sit between the two, or decay from one to the other.")
    print("\nOptimistic initialisation does poorly here, and the reason is")
    print("specific to this problem rather than a fault in the method: rewards")
    print("are binary and click-through rates are low, so a single impression")
    print("that fails to convert drags an arm's estimate from 0.10 to 0.00 and")
    print("the optimism is spent after one pull of each arm. Optimism works when")
    print("rewards are dense; with rare events an explicit epsilon does not.")


if __name__ == "__main__":
    main()
