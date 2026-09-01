"""
q_learning.py
------------------------------------------------------------------
Module 3 : Model-free control.  The agent learns Q(s,a) purely from
           simulated patient trajectories (no access to P or R), and the
           learned policy is compared with the DP optimal policy.

           Temporal-difference (sample-based Bellman) update:
             Q(s,a) <- Q(s,a) + alpha[ r + gamma*max_a' Q(s',a') - Q(s,a) ]
------------------------------------------------------------------
"""

import numpy as np
from patient_mdp import S, A, GAMMA, TERMINAL, step, STATE_LABEL, ACTION_LABEL

START_DIST = np.array([0.15, 0.25, 0.30, 0.20, 0.10, 0.0, 0.0])  # admissions
MAX_EPOCHS = 40                                                  # ~13 days


def sample_start(rng):
    return rng.choice(S, p=START_DIST)


def q_learning(episodes=30000, alpha=0.20, gamma=GAMMA,
               eps_start=1.0, eps_end=0.05, eps_decay=0.9997, seed=42,
               robbins_monro=True):
    """Off-policy TD control.

    With robbins_monro=True the learning rate for each state-action pair
    decays as alpha / (1 + 0.002 * N(s,a)), satisfying the stochastic-
    approximation conditions required for convergence to Q*.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((S, A))
    N = np.zeros((S, A))
    eps = eps_start
    ep_returns, ep_eps, ep_td = [], [], []

    for _ in range(episodes):
        s = sample_start(rng)
        total, td_sum, n = 0.0, 0.0, 0
        for _ in range(MAX_EPOCHS):
            a = rng.integers(A) if rng.random() < eps else int(Q[s].argmax())
            s2, r, done = step(s, a, rng)
            N[s, a] += 1
            lr = alpha / (1.0 + 0.002 * N[s, a]) if robbins_monro else alpha
            target = r + (0.0 if done else gamma * Q[s2].max())
            td = target - Q[s, a]
            Q[s, a] += lr * td
            td_sum += abs(td); n += 1
            total += r
            s = s2
            if done:
                break
        eps = max(eps_end, eps * eps_decay)
        ep_returns.append(total)
        ep_eps.append(eps)
        ep_td.append(td_sum / max(n, 1))

    policy = Q.argmax(axis=1)
    return Q, policy, np.array(ep_returns), np.array(ep_eps), np.array(ep_td)


def sarsa(episodes=30000, alpha=0.20, gamma=GAMMA,
          eps_start=1.0, eps_end=0.05, eps_decay=0.9997, seed=7):
    """On-policy control, used as a comparison baseline."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((S, A))
    N = np.zeros((S, A))
    eps = eps_start
    ep_returns = []

    def pick(s):
        return rng.integers(A) if rng.random() < eps else int(Q[s].argmax())

    for _ in range(episodes):
        s = sample_start(rng)
        a = pick(s)
        total = 0.0
        for _ in range(MAX_EPOCHS):
            s2, r, done = step(s, a, rng)
            a2 = pick(s2)
            N[s, a] += 1
            lr = alpha / (1.0 + 0.002 * N[s, a])
            target = r + (0.0 if done else gamma * Q[s2, a2])
            Q[s, a] += lr * (target - Q[s, a])
            total += r
            s, a = s2, a2
            if done:
                break
        eps = max(eps_end, eps * eps_decay)
        ep_returns.append(total)

    return Q, Q.argmax(axis=1), np.array(ep_returns)


if __name__ == "__main__":
    from dp_solver import value_iteration
    V_dp, pi_dp, Q_dp, _, _ = value_iteration()

    Q_ql, pi_ql, ret_ql, eps_ql, td_ql = q_learning()
    Q_sa, pi_sa, ret_sa = sarsa()

    non_term = [s for s in range(S) if s not in TERMINAL]
    agree_ql = np.mean([pi_ql[s] == pi_dp[s] for s in non_term]) * 100
    agree_sa = np.mean([pi_sa[s] == pi_dp[s] for s in non_term]) * 100

    print("State        DP-optimal        Q-learning        SARSA")
    for s in non_term:
        print(f"{STATE_LABEL[s]:<12} {ACTION_LABEL[pi_dp[s]]:<17}"
              f"{ACTION_LABEL[pi_ql[s]]:<18}{ACTION_LABEL[pi_sa[s]]}")
    print(f"\nPolicy agreement with DP optimum : Q-learning {agree_ql:.0f}% | "
          f"SARSA {agree_sa:.0f}%")
    print("Mean |Q_ql - Q_dp| over non-terminal states: "
          f"{np.abs(Q_ql[non_term]-Q_dp[non_term]).mean():.3f}")
