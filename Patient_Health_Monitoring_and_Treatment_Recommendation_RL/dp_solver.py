"""
dp_solver.py
------------------------------------------------------------------
Module 2 : Dynamic-Programming solution of the clinical MDP using the
           Bellman expectation and Bellman optimality equations.

           * Value Iteration   (Bellman optimality backup)
           * Policy Iteration  (evaluation + greedy improvement)
------------------------------------------------------------------
"""

import numpy as np
from patient_mdp import (S, A, P, R_EXPECTED, GAMMA, TERMINAL,
                         STATE_LABEL, ACTION_LABEL)


# ----------------------------------------------------------------- #
#  Bellman optimality :  V*(s) = max_a [ R(s,a) + g * S P(s'|s,a)V*(s') ]
# ----------------------------------------------------------------- #
def value_iteration(gamma=GAMMA, theta=1e-8, max_iter=1000, verbose=False):
    V = np.zeros(S)
    history = {"delta": [], "V": [], "policy": []}

    for k in range(1, max_iter + 1):
        V_old = V.copy()
        Q = R_EXPECTED + gamma * (P @ V_old)          # shape (S, A)
        V = Q.max(axis=1)
        for t in TERMINAL:                            # terminal states = 0
            V[t] = 0.0
        delta = np.abs(V - V_old).max()

        history["delta"].append(delta)
        history["V"].append(V.copy())
        history["policy"].append(Q.argmax(axis=1).copy())

        if verbose:
            print(f"iter {k:3d}  Bellman residual = {delta:.3e}")
        if delta < theta:
            break

    Q = R_EXPECTED + gamma * (P @ V)
    policy = Q.argmax(axis=1)
    return V, policy, Q, k, history


# ----------------------------------------------------------------- #
#  Bellman expectation :  V^p(s) = R(s,p(s)) + g * S P(s'|s,p(s))V^p(s')
# ----------------------------------------------------------------- #
def policy_evaluation(policy, gamma=GAMMA, theta=1e-10, max_iter=5000):
    V = np.zeros(S)
    for _ in range(max_iter):
        V_old = V.copy()
        for s in range(S):
            if s in TERMINAL:
                V[s] = 0.0
                continue
            a = policy[s]
            V[s] = R_EXPECTED[s, a] + gamma * P[s, a] @ V_old
        if np.abs(V - V_old).max() < theta:
            break
    return V


def policy_iteration(gamma=GAMMA, max_iter=100):
    policy = np.zeros(S, dtype=int)
    sweeps = 0
    history = []
    for k in range(1, max_iter + 1):
        V = policy_evaluation(policy, gamma)
        Q = R_EXPECTED + gamma * (P @ V)
        new_policy = Q.argmax(axis=1)
        history.append((policy.copy(), V.copy()))
        sweeps = k
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy, Q, sweeps, history


def policy_to_table(policy, V=None, Q=None):
    rows = []
    for s in range(S):
        if s in TERMINAL:
            rows.append((STATE_LABEL[s], "-- terminal --",
                         0.0 if V is None else V[s]))
        else:
            rows.append((STATE_LABEL[s], ACTION_LABEL[policy[s]],
                         None if V is None else V[s]))
    return rows


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    V_vi, pi_vi, Q_vi, k_vi, hist = value_iteration(verbose=False)
    V_pi, pi_pi, Q_pi, k_pi, _ = policy_iteration()

    print("=" * 66)
    print("VALUE ITERATION  - converged in", k_vi, "sweeps")
    print("=" * 66)
    for s in range(S):
        act = "-- terminal --" if s in TERMINAL else ACTION_LABEL[pi_vi[s]]
        print(f"  {STATE_LABEL[s]:<10}  V*(s) = {V_vi[s]:8.3f}   pi*(s) = {act}")

    print("\n" + "=" * 66)
    print("POLICY ITERATION - converged in", k_pi, "improvement steps")
    print("=" * 66)
    for s in range(S):
        act = "-- terminal --" if s in TERMINAL else ACTION_LABEL[pi_pi[s]]
        print(f"  {STATE_LABEL[s]:<10}  V*(s) = {V_pi[s]:8.3f}   pi*(s) = {act}")

    print("\nPolicies identical :", np.array_equal(pi_vi, pi_pi))
    print("max |V_VI - V_PI|  :", np.abs(V_vi - V_pi).max())

    print("\nOptimal action-value function Q*(s,a):")
    hdr = "  {:<10}".format("State") + "".join(f"{a:>14}" for a in ACTION_LABEL)
    print(hdr)
    for s in range(S):
        print(f"  {STATE_LABEL[s]:<10}" + "".join(f"{Q_vi[s,a]:14.3f}"
                                                 for a in range(A)))
