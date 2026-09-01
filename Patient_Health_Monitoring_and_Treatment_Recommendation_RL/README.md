# RL-Based Patient Health Monitoring and Treatment Recommendation System

**Course:** MLA0301 – Reinforcement Learning
**Topic:** Analysis and design of a reinforcement learning based patient health
monitoring and treatment recommendation system using Markov Decision Processes
and Bellman equations.

Patient health conditions are modelled as MDP states, clinical interventions as
actions and measured clinical outcomes as rewards. The Bellman optimality
equation is solved by dynamic programming to obtain an optimal treatment
policy, and the same policy is then recovered model-free from simulated
patient trajectories. The learned policy is analysed using treatment
effectiveness, recovery rate and cumulative reward.

---

## 1. Problem formulation

The ward process is modelled as `MDP = (S, A, P, R, gamma)` with one decision
epoch equal to an eight-hour clinical review window.

**States** — severity bands derived from a NEWS2-style score:

| State | Meaning | Type |
|---|---|---|
| S0 | Critical – organ instability | non-terminal |
| S1 | Severe – marked derangement | non-terminal |
| S2 | Moderate – active treatment required | non-terminal |
| S3 | Mild – responding to treatment | non-terminal |
| S4 | Stable – discharge can be considered | non-terminal |
| S5 | Recovered | terminal (absorbing) |
| S6 | Adverse outcome | terminal (absorbing) |

**Actions** — `a0` Monitor, `a1` Medication, `a2` Intensive therapy,
`a3` ICU escalation, `a4` Discharge with follow-up.

**Reward**

```
R(s,a) = w1*clinical_benefit - w2*treatment_cost
       - w3*adverse_risk     - w4*length_of_stay
```

with `w1=1.0, w2=0.6, w3=0.8, w4=1.0` and terminal outcomes of `+40` for
recovery and `-60` for an adverse event. The asymmetry is deliberate: it is
what makes the learned policy risk-averse rather than reward-seeking.

**Discount factor** — `gamma = 0.95`, giving an effective planning horizon of
about seven days at three decision epochs per day.

---

## 2. Repository layout

```
.
├── patient_mdp.py         MDP definition: S, A, P(s'|s,a), R(s,a), gamma,
│                          and the environment step() function
├── dp_solver.py           Value iteration and policy iteration
│                          (Bellman optimality / expectation equations)
├── q_learning.py          Model-free control: Q-learning and SARSA
├── evaluate.py            Monte-Carlo policy rollout, outcome metrics and
│                          the conventional baseline strategies
├── run_experiments.py     Master script — runs every experiment and writes
│                          all CSV / NPY result files
├── test_suite.py          15 verification test cases
├── figstyle.py            Shared plotting style and diagram primitives
├── make_diagrams.py       Architecture, MDP graph and flowchart figures
├── make_result_plots.py   Convergence, policy, learning-curve and
│                          comparison plots
├── results/               generated CSV / NPY output (git-ignored)
└── figures/               generated PNG figures (git-ignored)
```

---

## 3. Requirements

Python 3.10 or later.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. How to run

Run from inside this folder, in this order:

```bash
python patient_mdp.py         # build and validate the MDP
python dp_solver.py           # value iteration + policy iteration
python q_learning.py          # Q-learning and SARSA
python run_experiments.py     # all experiments -> results/
python test_suite.py          # 15 verification test cases
python make_diagrams.py       # figures 1-5  -> figures/
python make_result_plots.py   # figures 6-13 -> figures/
```

Every run is seeded, so the output is deterministic and reproducible.
Total runtime is under one minute on an ordinary laptop; no GPU is required.

---

## 5. Results

### Optimal policy

Solved by value iteration (56 sweeps, Bellman residual `6.11e-09`) and
independently confirmed by policy iteration (3 improvement steps, value
functions agreeing to `1.83e-08`).

| State | V*(s) | Recommended treatment | Margin over 2nd best |
|---|---|---|---|
| Critical | 24.10 | ICU escalation | 12.94 |
| Severe | 32.13 | Intensive therapy | 4.20 |
| Moderate | 42.57 | Standard medication | 6.30 |
| Mild | 44.00 | Standard medication | 2.84 |
| Stable | 44.92 | Discharge with follow-up | 1.50 |

The graded escalation ladder was not encoded by hand — it emerges from the
Bellman optimality equation given only the transition and reward models.

Q-learning and SARSA both recovered this policy exactly (100% agreement)
from 30 000 simulated episodes with no access to `P` or `R`.

### Policy benchmarking

5 000 simulated patients per policy:

| Policy | Recovery rate | Adverse rate | Cumulative reward | Treatment effectiveness |
|---|---|---|---|---|
| **RL optimal (value iteration)** | **88.40%** | **11.60%** | **44.42** | **54.02%** |
| RL learned (Q-learning) | 88.40% | 11.60% | 44.42 | 54.02% |
| Fixed clinical protocol | 77.32% | 21.82% | 32.48 | 53.08% |
| Uniform medication | 67.28% | 32.64% | 17.98 | 35.66% |
| Conservative monitoring | 22.18% | 76.52% | −45.02 | 65.92% |
| Aggressive escalation | 33.34% | 65.44% | −67.59 | 32.34% |
| Random policy | 32.10% | 67.90% | −40.13 | 34.62% |

Against the conventional fixed protocol the learned policy improves the
recovery rate by 11.08 points, nearly halves the adverse-outcome rate, and
does so while using 34.9% fewer interventions and 74.2% less critical-care
capacity per patient.

Note that conservative monitoring records the highest raw treatment
effectiveness while being by far the worst policy — it only ever intervenes
once a patient is already critical, so its denominator is tiny. This is why
the three metrics are reported together rather than optimised individually.

Repeated over ten independent seeds, the RL recovery rate is
`88.86 ± 0.59%` against `76.84 ± 0.63%` for the fixed protocol.

### Verification

`test_suite.py` reports 15/15 passing, covering model validity
(row-stochastic transitions, absorbing terminals), algorithmic correctness
(convergence, cross-algorithm agreement, Bellman residual) and clinical
safety (no unstable patient is ever discharged).

---

## 6. Scope and limitations

The environment is a clinically parameterised simulator. Transition
probabilities are elicited from clinical plausibility rules rather than
estimated from a patient registry, so **all numerical results are
proof-of-concept values and not clinical evidence.** No real or identifiable
patient data is used anywhere in this repository.

Other known limitations: seven severity bands cannot represent comorbidity,
age or trajectory; the Markov assumption ignores rate of change; the five
actions are coarse relative to real drug, dose and timing decisions; and the
reward weights encode a value judgement rather than a measurement.

Any real deployment would require the transition model to be estimated from
de-identified records, off-policy evaluation before prospective use, and a
mandatory clinician-approval step — the agent recommends, it never acts.

---

## 7. References

1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*,
   2nd ed., MIT Press, 2018.
2. M. L. Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic
   Programming*, Wiley, 1994.
3. C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*,
   vol. 8, pp. 279–292, 1992.
4. M. Komorowski et al., "The Artificial Intelligence Clinician learns optimal
   treatment strategies for sepsis in intensive care," *Nature Medicine*,
   vol. 24, pp. 1716–1720, 2018.
5. O. Gottesman et al., "Guidelines for reinforcement learning in healthcare,"
   *Nature Medicine*, vol. 25, pp. 16–18, 2019.
6. Royal College of Physicians, *National Early Warning Score (NEWS) 2*,
   RCP London, 2017.
