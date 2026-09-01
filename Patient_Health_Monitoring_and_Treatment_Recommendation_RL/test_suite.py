"""
test_suite.py
------------------------------------------------------------------
Module 6 : Verification test cases for the clinical RL system.
Run:  python3 test_suite.py
------------------------------------------------------------------
"""

import numpy as np
from patient_mdp import (S, A, P, R_EXPECTED, GAMMA, TERMINAL, STATE_LABEL,
                         ACTION_LABEL, step)
from dp_solver import value_iteration, policy_iteration, policy_evaluation
from q_learning import q_learning
from evaluate import rollout, build_baselines

RESULTS = []


def check(tid, name, expected, actual, ok):
    RESULTS.append((tid, name, expected, actual, "PASS" if ok else "FAIL"))


V, PIS, Q, K, _ = value_iteration()
Vp, PIp, Qp, Kp, _ = policy_iteration()

# TC-01 transition matrix is row-stochastic
rs = P.sum(axis=2)
check("TC-01", "Every P(.|s,a) row sums to 1",
      "all rows = 1.000",
      f"min {rs.min():.6f}, max {rs.max():.6f}",
      np.allclose(rs, 1.0))

# TC-02 no negative probabilities
check("TC-02", "All transition probabilities are non-negative",
      "min >= 0", f"min = {P.min():.4f}", P.min() >= 0)

# TC-03 terminal states are absorbing
absorb = all(np.isclose(P[t, a, t], 1.0) for t in TERMINAL for a in range(A))
check("TC-03", "Terminal states S5/S6 are absorbing",
      "P(t|t,a) = 1 for all a", "satisfied" if absorb else "violated", absorb)

# TC-04 value iteration converges
check("TC-04", "Value iteration converges below tolerance 1e-8",
      "converges in < 1000 sweeps", f"converged in {K} sweeps", K < 1000)

# TC-05 VI and PI agree on the policy
same = np.array_equal(PIS, PIp)
check("TC-05", "Value iteration and policy iteration give the same policy",
      "policies identical", "identical" if same else "different", same)

# TC-06 VI and PI agree on the value function
gap = float(np.abs(V - Vp).max())
check("TC-06", "Value functions of VI and PI agree",
      "max |V_VI - V_PI| < 1e-6", f"{gap:.2e}", gap < 1e-6)

# TC-07 Bellman optimality residual of the returned V* is ~0
resid = float(np.abs(V - np.where(np.arange(S) < 5,
                                  (R_EXPECTED + GAMMA * (P @ V)).max(axis=1),
                                  0.0)).max())
check("TC-07", "V* satisfies the Bellman optimality equation",
      "residual < 1e-6", f"{resid:.2e}", resid < 1e-6)

# TC-08 clinical sanity - critical patients are escalated, never discharged
crit_ok = (ACTION_LABEL[PIS[0]] == "ICU Escalation")
check("TC-08", "Critical patient is escalated to critical care",
      "pi*(Critical) = ICU Escalation",
      f"pi*(Critical) = {ACTION_LABEL[PIS[0]]}", crit_ok)

# TC-09 clinical sanity - no unstable patient is discharged
unsafe = [STATE_LABEL[s] for s in range(5)
          if ACTION_LABEL[PIS[s]] == "Discharge" and s < 4]
check("TC-09", "No unstable patient (S0-S3) is discharged",
      "empty set", f"{unsafe if unsafe else 'none'}", len(unsafe) == 0)

# TC-10 value ordering - healthier states are worth more
mono = all(V[i] < V[i + 1] for i in range(4))
check("TC-10", "V* increases monotonically with clinical improvement",
      "V*(S0) < ... < V*(S4)",
      " < ".join(f"{V[i]:.1f}" for i in range(5)), mono)

# TC-11 model-free learner recovers the DP-optimal policy
_, PIQ, _, _, _ = q_learning(episodes=30000)
nt = [s for s in range(S) if s not in TERMINAL]
agree = float(np.mean([PIQ[s] == PIS[s] for s in nt]) * 100)
check("TC-11", "Q-learning recovers the DP-optimal policy",
      "agreement >= 80%", f"{agree:.0f}% agreement", agree >= 80)

# TC-12 learned policy beats every clinical baseline on recovery rate
m_rl, _, _ = rollout(PIS, episodes=3000, seed=999)
base = build_baselines()
best_base = max(rollout(p, episodes=3000, seed=999)[0]["Recovery Rate (%)"]
                for p in base.values())
check("TC-12", "RL policy outperforms all baselines on recovery rate",
      f"RL > {best_base:.1f}%",
      f"RL = {m_rl['Recovery Rate (%)']:.1f}%",
      m_rl["Recovery Rate (%)"] > best_base)

# TC-13 discounted return is finite and bounded
bound = (np.abs(R_EXPECTED).max() + 60) / (1 - GAMMA)
check("TC-13", "V* is bounded by Rmax/(1-gamma)",
      f"|V*| <= {bound:.1f}", f"max |V*| = {np.abs(V).max():.2f}",
      np.abs(V).max() <= bound)

# TC-14 simulator honours terminal absorption
rng = np.random.default_rng(5)
s2, r, done = step(5, 2, rng)
check("TC-14", "Simulator keeps a recovered patient in the terminal state",
      "next state = S5_Recovered, done = True",
      f"next = {STATE_LABEL[s2]}, done = {done}", s2 == 5 and done)

# TC-15 reproducibility: identical seed gives identical metrics
a1, _, _ = rollout(PIS, episodes=1500, seed=321)
a2, _, _ = rollout(PIS, episodes=1500, seed=321)
same_run = all(abs(a1[k] - a2[k]) < 1e-12 for k in a1 if a1[k] == a1[k])
check("TC-15", "Evaluation is reproducible for a fixed random seed",
      "identical metrics", "identical" if same_run else "differs", same_run)


if __name__ == "__main__":
    w = (7, 58, 34, 34, 6)
    line = "-" * (sum(w) + 8)
    print(line)
    print(f"{'ID':<7}{'Test case':<58}{'Expected':<34}"
          f"{'Actual':<34}{'Result':<6}")
    print(line)
    for r in RESULTS:
        print(f"{r[0]:<7}{r[1]:<58}{r[2]:<34}{r[3]:<34}{r[4]:<6}")
    print(line)
    npass = sum(1 for r in RESULTS if r[4] == "PASS")
    print(f"{npass}/{len(RESULTS)} test cases passed")

    import csv
    with open("results/test_cases.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Test ID", "Test Case", "Expected Result",
                       "Actual Result", "Status"])
        wcsv.writerows(RESULTS)
