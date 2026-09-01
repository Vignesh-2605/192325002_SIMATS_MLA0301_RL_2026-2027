"""
evaluate.py
------------------------------------------------------------------
Module 4 : Monte-Carlo policy evaluation and clinical benchmarking.

Every policy is rolled out over the same number of simulated patient
episodes and scored on the three metrics required by the assignment
(treatment effectiveness, recovery rate, cumulative reward) plus the
supporting clinical / operational indicators.
------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from patient_mdp import (S, A, GAMMA, TERMINAL, step, STATE_LABEL,
                         ACTION_LABEL)
from q_learning import START_DIST, MAX_EPOCHS

EPOCHS_PER_DAY = 3                       # one decision epoch = 8 hours
# Higher rank = better clinical condition; adverse terminal is the worst.
RANK = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: -1}
ACTIVE_TREATMENTS = {1, 2, 3}            # Medication / Intensive / ICU


def rollout(policy, episodes=5000, seed=2026, gamma=GAMMA):
    """Simulate `episodes` patients under a fixed deterministic policy."""
    rng = np.random.default_rng(seed)
    rec = dict(recovered=0, adverse=0, timeout=0,
               ret=[], disc=[], los=[], los_rec=[],
               improve=0, interventions=0, icu_epochs=0,
               action_counts=np.zeros(A), visits=np.zeros(S))

    traj_states = np.full((episodes, MAX_EPOCHS + 1), -1, dtype=int)

    for e in range(episodes):
        s = rng.choice(S, p=START_DIST)
        total, disc, t = 0.0, 0.0, 0
        traj_states[e, 0] = s
        for t in range(1, MAX_EPOCHS + 1):
            a = int(policy[s])
            rec["action_counts"][a] += 1
            rec["visits"][s] += 1
            if a in ACTIVE_TREATMENTS:
                rec["interventions"] += 1
            if a == 3:
                rec["icu_epochs"] += 1
            s2, r, done = step(s, a, rng)
            if a in ACTIVE_TREATMENTS and RANK[s2] > RANK[s]:
                rec["improve"] += 1
            total += r
            disc += (gamma ** (t - 1)) * r
            traj_states[e, t] = s2
            s = s2
            if done:
                break
        rec["ret"].append(total)
        rec["disc"].append(disc)
        rec["los"].append(t)
        if s == 5:
            rec["recovered"] += 1
            rec["los_rec"].append(t)
        elif s == 6:
            rec["adverse"] += 1
        else:
            rec["timeout"] += 1

    n = episodes
    out = {
        "Recovery Rate (%)": 100.0 * rec["recovered"] / n,
        "Adverse Outcome Rate (%)": 100.0 * rec["adverse"] / n,
        "Unresolved at Horizon (%)": 100.0 * rec["timeout"] / n,
        "Mean Cumulative Reward": float(np.mean(rec["ret"])),
        "Std Cumulative Reward": float(np.std(rec["ret"])),
        "Mean Discounted Return": float(np.mean(rec["disc"])),
        "Treatment Effectiveness (%)":
            100.0 * rec["improve"] / max(rec["interventions"], 1),
        "Mean Time to Recovery (days)":
            float(np.mean(rec["los_rec"])) / EPOCHS_PER_DAY
            if rec["los_rec"] else float("nan"),
        "Mean Length of Stay (days)":
            float(np.mean(rec["los"])) / EPOCHS_PER_DAY,
        "ICU Epochs per Patient": rec["icu_epochs"] / n,
        "Interventions per Patient": rec["interventions"] / n,
    }
    return out, rec, traj_states


# ----------------------------------------------------------------- #
#  Baseline (non-learning) clinical policies                          #
# ----------------------------------------------------------------- #
def build_baselines():
    """Return {name: policy_vector}."""
    b = {}
    # Conventional severity-threshold protocol used in many wards
    b["Fixed Clinical Protocol"] = np.array([3, 3, 2, 1, 0, 0, 0])
    # Uniform pharmacotherapy for every admitted patient
    b["Uniform Medication"] = np.array([1, 1, 1, 1, 1, 0, 0])
    # Conservative watch-and-wait, escalating only when critical
    b["Conservative Monitoring"] = np.array([3, 0, 0, 0, 0, 0, 0])
    # Aggressive treatment of everybody
    b["Aggressive Escalation"] = np.array([3, 3, 3, 3, 3, 0, 0])
    return b


def evaluate_all(pi_dp, pi_ql, episodes=5000):
    policies = {"RL Optimal Policy (Value Iteration)": pi_dp,
                "RL Learned Policy (Q-Learning)": pi_ql}
    policies.update(build_baselines())

    rng = np.random.default_rng(11)
    policies["Random Policy"] = rng.integers(0, A, size=S)

    rows, extras = {}, {}
    for name, pol in policies.items():
        m, rec, traj = rollout(pol, episodes=episodes)
        rows[name] = m
        extras[name] = (rec, traj)
    df = pd.DataFrame(rows).T
    return df, extras, policies


if __name__ == "__main__":
    from dp_solver import value_iteration
    from q_learning import q_learning

    _, pi_dp, _, _, _ = value_iteration()
    _, pi_ql, _, _, _ = q_learning()

    df, extras, pols = evaluate_all(pi_dp, pi_ql, episodes=5000)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.round(2).to_string())
    df.round(3).to_csv("results/policy_comparison.csv")
