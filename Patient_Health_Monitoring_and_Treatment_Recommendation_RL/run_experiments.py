"""
run_experiments.py
------------------------------------------------------------------
Module 5 : Master experiment script.  Executes every experiment
reported in the assignment and writes all numerical results to
results/*.csv and results/summary.json
------------------------------------------------------------------
"""

import json
import numpy as np
import pandas as pd

from patient_mdp import (S, A, P, R_EXPECTED, GAMMA, TERMINAL,
                         STATE_LABEL, ACTION_LABEL)
from dp_solver import value_iteration, policy_iteration, policy_evaluation
from q_learning import q_learning, sarsa
from evaluate import evaluate_all, rollout, build_baselines

RES = "results/"
np.random.seed(0)
out = {}

# ---------------------------------------------------------------- #
# EXP-1  Dynamic programming: value iteration vs policy iteration    #
# ---------------------------------------------------------------- #
V_vi, pi_vi, Q_vi, k_vi, hist = value_iteration()
V_pi, pi_pi, Q_pi, k_pi, hist_pi = policy_iteration()

out["value_iteration_sweeps"] = int(k_vi)
out["policy_iteration_steps"] = int(k_pi)
out["policies_identical"] = bool(np.array_equal(pi_vi, pi_pi))
out["max_value_gap_vi_pi"] = float(np.abs(V_vi - V_pi).max())

pd.DataFrame({
    "State": STATE_LABEL,
    "V*(s)": np.round(V_vi, 3),
    "Optimal Action pi*(s)": [ACTION_LABEL[a] if s not in TERMINAL
                              else "terminal" for s, a in enumerate(pi_vi)],
}).to_csv(RES + "optimal_policy.csv", index=False)

pd.DataFrame(np.round(Q_vi, 3), index=STATE_LABEL,
             columns=ACTION_LABEL).to_csv(RES + "q_star.csv")

np.save(RES + "vi_history_delta.npy", np.array(hist["delta"]))
np.save(RES + "vi_history_V.npy", np.array(hist["V"]))

# ---------------------------------------------------------------- #
# EXP-2  Model-free learning                                        #
# ---------------------------------------------------------------- #
Q_ql, pi_ql, ret_ql, eps_ql, td_ql = q_learning(episodes=30000)
Q_sa, pi_sa, ret_sa = sarsa(episodes=30000)

non_term = [s for s in range(S) if s not in TERMINAL]
out["q_learning_policy_agreement_pct"] = float(
    np.mean([pi_ql[s] == pi_vi[s] for s in non_term]) * 100)
out["sarsa_policy_agreement_pct"] = float(
    np.mean([pi_sa[s] == pi_vi[s] for s in non_term]) * 100)
out["q_learning_rmse_vs_Qstar"] = float(
    np.sqrt(((Q_ql[non_term] - Q_vi[non_term]) ** 2).mean()))
out["sarsa_rmse_vs_Qstar"] = float(
    np.sqrt(((Q_sa[non_term] - Q_vi[non_term]) ** 2).mean()))

np.save(RES + "ql_returns.npy", ret_ql)
np.save(RES + "ql_eps.npy", eps_ql)
np.save(RES + "ql_td.npy", td_ql)
np.save(RES + "sarsa_returns.npy", ret_sa)
np.save(RES + "Q_ql.npy", Q_ql)
np.save(RES + "Q_star.npy", Q_vi)
np.save(RES + "V_star.npy", V_vi)
np.save(RES + "pi_star.npy", pi_vi)
np.save(RES + "pi_ql.npy", pi_ql)

pd.DataFrame({
    "State": [STATE_LABEL[s] for s in non_term],
    "DP Optimal": [ACTION_LABEL[pi_vi[s]] for s in non_term],
    "Q-Learning": [ACTION_LABEL[pi_ql[s]] for s in non_term],
    "SARSA": [ACTION_LABEL[pi_sa[s]] for s in non_term],
    "Match": ["Yes" if pi_ql[s] == pi_vi[s] else "No" for s in non_term],
}).to_csv(RES + "policy_agreement.csv", index=False)

# ---------------------------------------------------------------- #
# EXP-3  Policy benchmarking against clinical baselines             #
# ---------------------------------------------------------------- #
df_cmp, extras, policies = evaluate_all(pi_vi, pi_ql, episodes=5000)
df_cmp.round(3).to_csv(RES + "policy_comparison.csv")
out["comparison"] = json.loads(df_cmp.round(3).to_json(orient="index"))

# ---------------------------------------------------------------- #
# EXP-4  Repeated-seed robustness (95% CI on the headline metrics)  #
# ---------------------------------------------------------------- #
seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
robust = {}
for name in ["RL Optimal Policy (Value Iteration)",
             "Fixed Clinical Protocol", "Uniform Medication"]:
    pol = policies[name]
    rr, cr, te = [], [], []
    for sd in seeds:
        m, _, _ = rollout(pol, episodes=2000, seed=sd)
        rr.append(m["Recovery Rate (%)"])
        cr.append(m["Mean Cumulative Reward"])
        te.append(m["Treatment Effectiveness (%)"])
    robust[name] = {
        "recovery_mean": float(np.mean(rr)),
        "recovery_ci95": float(1.96 * np.std(rr, ddof=1) / np.sqrt(len(seeds))),
        "reward_mean": float(np.mean(cr)),
        "reward_ci95": float(1.96 * np.std(cr, ddof=1) / np.sqrt(len(seeds))),
        "effectiveness_mean": float(np.mean(te)),
        "effectiveness_ci95": float(1.96 * np.std(te, ddof=1) / np.sqrt(len(seeds))),
    }
out["robustness"] = robust
pd.DataFrame(robust).T.round(3).to_csv(RES + "robustness_ci.csv")

# ---------------------------------------------------------------- #
# EXP-5  Discount-factor sensitivity                                #
# ---------------------------------------------------------------- #
gammas = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
sens_rows = []
for g in gammas:
    Vg, pig, Qg, kg, _ = value_iteration(gamma=g)
    sens_rows.append({
        "gamma": g,
        "sweeps": kg,
        **{STATE_LABEL[s]: ACTION_LABEL[pig[s]] for s in non_term},
        "V*(Critical)": round(float(Vg[0]), 2),
        "V*(Stable)": round(float(Vg[4]), 2),
    })
df_sens = pd.DataFrame(sens_rows)
df_sens.to_csv(RES + "gamma_sensitivity.csv", index=False)
out["gamma_sensitivity"] = json.loads(df_sens.to_json(orient="records"))

# ---------------------------------------------------------------- #
# EXP-6  Per-state treatment effectiveness under the optimal policy #
# ---------------------------------------------------------------- #
RANKS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: -1}
te_rows = []
for s in non_term:
    a = int(pi_vi[s])
    p_improve = sum(P[s, a, sp] for sp in range(S) if RANKS[sp] > RANKS[s])
    p_same = sum(P[s, a, sp] for sp in range(S) if RANKS[sp] == RANKS[s])
    p_worse = sum(P[s, a, sp] for sp in range(S) if RANKS[sp] < RANKS[s])
    te_rows.append({
        "State": STATE_LABEL[s],
        "Recommended Action": ACTION_LABEL[a],
        "P(improve)": round(float(p_improve), 3),
        "P(unchanged)": round(float(p_same), 3),
        "P(deteriorate)": round(float(p_worse), 3),
        "Q*(s,a)": round(float(Q_vi[s, a]), 2),
        "Advantage over 2nd best": round(float(
            Q_vi[s, a] - np.sort(Q_vi[s])[-2]), 2),
    })
df_te = pd.DataFrame(te_rows)
df_te.to_csv(RES + "treatment_effectiveness.csv", index=False)
out["treatment_effectiveness"] = json.loads(df_te.to_json(orient="records"))

# ---------------------------------------------------------------- #
# EXP-7  State-occupancy trajectories (RL vs Fixed protocol)        #
# ---------------------------------------------------------------- #
occ = {}
for name in ["RL Optimal Policy (Value Iteration)", "Fixed Clinical Protocol"]:
    _, _, traj = rollout(policies[name], episodes=4000, seed=77)
    T = 16
    mat = np.zeros((T, S))
    for t in range(T):
        col = traj[:, t]
        # once absorbed, patients remain in the terminal state
        col = np.where(col == -1, traj[:, :t + 1].max(axis=1), col)
        for s in range(S):
            mat[t, s] = np.mean(col == s)
    occ[name] = mat.tolist()
    np.save(RES + f"occupancy_{name.split()[0]}.npy", mat)
out["occupancy_states"] = STATE_LABEL

with open(RES + "summary.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps({k: v for k, v in out.items()
                  if k not in ("comparison", "gamma_sensitivity",
                               "treatment_effectiveness", "robustness")},
                 indent=2))
print("\n--- Policy comparison ---")
print(df_cmp.round(2).to_string())
print("\n--- Treatment effectiveness per state ---")
print(df_te.to_string(index=False))
print("\n--- Gamma sensitivity ---")
print(df_sens.to_string(index=False))
print("\n--- Robustness (95% CI) ---")
print(pd.DataFrame(robust).T.round(2).to_string())
