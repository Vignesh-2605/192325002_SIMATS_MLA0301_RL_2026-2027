"""
patient_mdp.py
------------------------------------------------------------------
MLA03 - Reinforcement Learning
RL-Based Patient Health Monitoring and Treatment Recommendation System

Module 1 : Markov Decision Process definition
            MDP = (S, A, P, R, gamma)

The MDP encodes a clinical deterioration / recovery process for an
in-patient under continuous monitoring.  Each decision epoch equals one
clinical review window (8 hours).
------------------------------------------------------------------
"""

import numpy as np

# ----------------------------------------------------------------- #
# 1. STATE SPACE  S                                                  #
# ----------------------------------------------------------------- #
# Non-terminal clinical states are ordered by severity (0 = worst).
STATES = [
    "S0_Critical",      # organ instability, NEWS2 >= 7
    "S1_Severe",        # marked physiological derangement, NEWS2 5-6
    "S2_Moderate",      # moderate derangement, NEWS2 3-4
    "S3_Mild",          # mild derangement, NEWS2 1-2
    "S4_Stable",        # physiologically stable, NEWS2 0
    "S5_Recovered",     # TERMINAL - discharged, treatment successful
    "S6_Adverse",       # TERMINAL - death / irreversible deterioration
]
S = len(STATES)
TERMINAL = {5, 6}

# Short labels used in figures and tables
STATE_LABEL = ["Critical", "Severe", "Moderate", "Mild", "Stable",
               "Recovered", "Adverse"]

# ----------------------------------------------------------------- #
# 2. ACTION SPACE  A                                                 #
# ----------------------------------------------------------------- #
ACTIONS = [
    "A0_Monitor",       # continue passive monitoring only
    "A1_Medication",    # standard pharmacological therapy
    "A2_Intensive",     # intensive / combination pharmacotherapy
    "A3_ICU",           # critical-care escalation (ventilation/vasopressor)
    "A4_Discharge",     # discharge with tele-follow-up
]
A = len(ACTIONS)
ACTION_LABEL = ["Monitor", "Medication", "Intensive", "ICU Escalation",
                "Discharge"]

GAMMA = 0.95            # discount factor - long-horizon clinical outcome

# ----------------------------------------------------------------- #
# 3. TRANSITION MODEL  P(s' | s, a)                                  #
# ----------------------------------------------------------------- #
# Each row is a probability distribution over the 7 states, elicited from
# clinical-plausibility rules:
#   * doing nothing in a bad state -> high probability of deterioration
#   * escalation helps most in severe states but carries iatrogenic risk
#   * over-treating a stable patient wastes resources and adds risk
#   * discharging an unstable patient is dangerous (readmission / adverse)
#
# Index order: [Critical, Severe, Moderate, Mild, Stable, Recovered, Adverse]

P = np.zeros((S, A, S))

# ---------------- S0 : Critical ---------------------------------- #
P[0, 0] = [0.30, 0.05, 0.00, 0.00, 0.00, 0.00, 0.65]   # Monitor only - fatal
P[0, 1] = [0.45, 0.20, 0.02, 0.00, 0.00, 0.00, 0.33]   # Medication - too weak
P[0, 2] = [0.35, 0.34, 0.10, 0.01, 0.00, 0.00, 0.20]   # Intensive therapy
P[0, 3] = [0.24, 0.42, 0.19, 0.04, 0.01, 0.00, 0.10]   # ICU escalation - best
P[0, 4] = [0.02, 0.03, 0.00, 0.00, 0.00, 0.05, 0.90]   # Discharge - unsafe

# ---------------- S1 : Severe ------------------------------------ #
P[1, 0] = [0.42, 0.33, 0.05, 0.00, 0.00, 0.00, 0.20]
P[1, 1] = [0.20, 0.40, 0.28, 0.05, 0.01, 0.00, 0.06]
P[1, 2] = [0.10, 0.30, 0.38, 0.13, 0.03, 0.00, 0.06]   # Intensive - best
P[1, 3] = [0.09, 0.33, 0.34, 0.11, 0.03, 0.00, 0.10]   # ICU - similar, costly
P[1, 4] = [0.30, 0.15, 0.03, 0.00, 0.00, 0.10, 0.42]

# ---------------- S2 : Moderate ---------------------------------- #
P[2, 0] = [0.08, 0.32, 0.42, 0.13, 0.03, 0.00, 0.02]
P[2, 1] = [0.02, 0.10, 0.36, 0.38, 0.12, 0.01, 0.01]   # Medication - best
P[2, 2] = [0.02, 0.09, 0.33, 0.36, 0.14, 0.02, 0.04]   # Intensive - toxicity
P[2, 3] = [0.03, 0.12, 0.38, 0.30, 0.09, 0.01, 0.07]   # ICU - over-treatment
P[2, 4] = [0.10, 0.25, 0.28, 0.05, 0.02, 0.22, 0.08]

# ---------------- S3 : Mild -------------------------------------- #
P[3, 0] = [0.00, 0.05, 0.22, 0.48, 0.22, 0.03, 0.00]
P[3, 1] = [0.00, 0.02, 0.08, 0.34, 0.44, 0.11, 0.01]   # Medication - best
P[3, 2] = [0.00, 0.03, 0.10, 0.33, 0.39, 0.10, 0.05]
P[3, 3] = [0.01, 0.05, 0.16, 0.38, 0.30, 0.04, 0.06]
P[3, 4] = [0.00, 0.03, 0.12, 0.20, 0.15, 0.47, 0.03]

# ---------------- S4 : Stable ------------------------------------ #
P[4, 0] = [0.00, 0.01, 0.05, 0.20, 0.62, 0.12, 0.00]
P[4, 1] = [0.00, 0.01, 0.03, 0.14, 0.60, 0.21, 0.01]
P[4, 2] = [0.00, 0.02, 0.05, 0.16, 0.53, 0.20, 0.04]
P[4, 3] = [0.01, 0.03, 0.08, 0.20, 0.55, 0.08, 0.05]
P[4, 4] = [0.00, 0.00, 0.02, 0.06, 0.10, 0.81, 0.01]   # Discharge - best

# ---------------- Terminal states are absorbing ------------------ #
for a in range(A):
    P[5, a] = np.eye(S)[5]
    P[6, a] = np.eye(S)[6]

# Numerical safety: renormalise every row
for s in range(S):
    for a in range(A):
        P[s, a] = P[s, a] / P[s, a].sum()

# ----------------------------------------------------------------- #
# 4. REWARD MODEL  R(s, a)                                           #
# ----------------------------------------------------------------- #
# R(s,a) = w1 * clinical_benefit  - w2 * treatment_cost
#          - w3 * adverse_risk    - w4 * length_of_stay_penalty
#
# Component matrices (per decision epoch), all on a 0-10 scale.

# Clinical benefit obtained by applying action a while in state s
CLINICAL_BENEFIT = np.array([
    # Mon  Med  Int  ICU  Dis
    [0.0, 2.0, 5.0, 8.0, 0.0],   # Critical
    [0.5, 4.0, 6.5, 6.0, 0.0],   # Severe
    [1.0, 6.0, 5.0, 3.5, 2.0],   # Moderate
    [2.0, 5.5, 4.0, 2.0, 5.0],   # Mild
    [3.0, 3.0, 2.0, 1.0, 7.0],   # Stable
    [0.0, 0.0, 0.0, 0.0, 0.0],   # Recovered  (terminal)
    [0.0, 0.0, 0.0, 0.0, 0.0],   # Adverse    (terminal)
])

# Resource / economic cost of the action (independent of state)
TREATMENT_COST = np.array([0.2, 1.5, 3.5, 7.0, 0.5])

# Iatrogenic / adverse-event risk of applying action a in state s
ADVERSE_RISK = np.array([
    [6.0, 3.0, 2.0, 1.0, 9.0],   # Critical  (monitoring & discharge unsafe)
    [4.0, 1.5, 1.5, 2.0, 7.0],   # Severe
    [2.0, 0.5, 2.0, 3.0, 3.5],   # Moderate
    [1.0, 0.3, 1.5, 3.0, 1.0],   # Mild
    [0.5, 0.5, 1.5, 3.0, 0.2],   # Stable
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
])

# Length-of-stay penalty: every epoch the patient remains admitted
LOS_PENALTY = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

W1, W2, W3, W4 = 1.0, 0.6, 0.8, 1.0

R = np.zeros((S, A))
for s in range(S):
    for a in range(A):
        R[s, a] = (W1 * CLINICAL_BENEFIT[s, a]
                   - W2 * TREATMENT_COST[a]
                   - W3 * ADVERSE_RISK[s, a]
                   - W4 * LOS_PENALTY[s])

# Terminal outcome rewards (received on entering the terminal state)
TERMINAL_REWARD = {5: +40.0, 6: -60.0}
for a in range(A):
    R[5, a] = 0.0
    R[6, a] = 0.0

# Expected one-step reward including the terminal bonus/penalty
R_EXPECTED = np.zeros((S, A))
for s in range(S):
    for a in range(A):
        bonus = sum(P[s, a, sp] * TERMINAL_REWARD.get(sp, 0.0)
                    for sp in range(S))
        R_EXPECTED[s, a] = 0.0 if s in TERMINAL else R[s, a] + bonus


def step(s, a, rng):
    """Sample one environment transition. Returns (s_next, reward, done)."""
    s_next = rng.choice(S, p=P[s, a])
    r = 0.0 if s in TERMINAL else R[s, a] + TERMINAL_REWARD.get(s_next, 0.0)
    return s_next, r, s_next in TERMINAL


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("States :", STATES)
    print("Actions:", ACTIONS)
    print("\nExpected immediate reward R(s,a):\n", R_EXPECTED)
    print("\nRow-sum check:", np.allclose(P.sum(axis=2), 1.0))
