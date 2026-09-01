"""Structural diagrams: architecture, MDP graph, flowcharts, agent loop."""
import numpy as np
from figstyle import *

FIG = "figures/"
g, pi, sig, ein, ra, la = G_GAMMA, G_PI, G_SIGMA, G_IN, G_RARR, G_LARR
th, eps, pr = G_THETA, G_EPS, G_PRIME
a_ = [f"a{SUB[i]}" for i in range(5)]
s_t = f"s{SUB[0]}"


# ================================================================= #
# FIG 1 - System architecture (closed clinical decision loop)        #
# ================================================================= #
def fig_architecture():
    fig, ax = canvas(10.4, 6.1, (0, 104), (2, 65),
                     "Figure 1   System Architecture of the RL-Based Patient "
                     "Monitoring and Treatment Recommendation System")

    box(ax, 3, 44, 18, 12,
        f"1.  CLINICAL DATA\nACQUISITION\n\n{G_BULL} Bedside monitors: HR, BP,\n"
        f"   SpO₂, RR, temperature\n{G_BULL} Laboratory results, EHR",
        fc=LBLUE, fs=7.0)
    box(ax, 24, 44, 18, 12,
        f"2.  PRE-PROCESSING\n\n{G_BULL} Missing-value imputation\n"
        f"{G_BULL} Noise / artefact filtering\n{G_BULL} Normalisation, windowing",
        fc=LBLUE, fs=7.0)
    box(ax, 45, 44, 18, 12,
        f"3.  STATE ABSTRACTION\n\n{G_BULL} NEWS2 severity scoring\n"
        f"{G_BULL} Discretisation into\n   s {ein} S = {{S0 ... S6}}",
        fc=LBLUE, fs=7.0)
    box(ax, 66, 44, 18, 12,
        f"4.  MDP MODEL BUILDER\n\n{G_BULL} Transition P(s{pr}|s,a)\n"
        f"{G_BULL} Reward model R(s,a)\n{G_BULL} Discount factor {g}",
        fc=LPURPLE, fs=7.0)

    ax.text(63, 41.0, f"MDP = (S, A, P, R, {g})", ha="center", fontsize=7.8,
            color=PURPLE, fontweight="bold")

    # RL agent panel
    box(ax, 22, 22, 60, 15, "", fc="#fbfbfb", ec=GREY, lw=1.3)
    ax.text(52, 34.6, "5.   REINFORCEMENT LEARNING AGENT   "
                      "(Bellman optimality core)",
            ha="center", va="center", fontsize=8.6, fontweight="bold")
    box(ax, 24.5, 24, 17, 8.4,
        f"Value Iteration\nV(s) {la} maxₐ [R + {g}{sig}PV]", fc=LGREEN, fs=7.2)
    box(ax, 43.5, 24, 17, 8.4,
        "Policy Iteration\nevaluate " + ra + " improve", fc=LGREEN, fs=7.2)
    box(ax, 62.5, 24, 17, 8.4,
        "Q-Learning (TD)\nmodel-free control", fc=LGREEN, fs=7.2)

    box(ax, 3, 6, 19, 11,
        f"6.  OPTIMAL POLICY\n{pi}*(s) = arg maxₐ Q*(s,a)\n\n"
        "Treatment recommendation", fc=LORANGE, fs=7.2)
    box(ax, 25, 6, 19, 11,
        "7.  CLINICIAN REVIEW\n(human-in-the-loop)\n\n"
        "accept / override / defer", fc=LORANGE, fs=7.2)
    box(ax, 47, 6, 19, 11,
        "8.  INTERVENTION\nDELIVERY\n\nmedication | intensive |\n"
        "ICU | discharge", fc=LORANGE, fs=7.2)
    box(ax, 69, 6, 19, 11,
        "9.  OUTCOME MONITOR\n\nreward r " + f"computed from\n"
        "the observed clinical\nresponse", fc=LRED, fs=7.2)

    for x0, x1 in [(21, 24), (42, 45), (63, 66)]:
        arrow(ax, (x0, 50), (x1, 50), lw=1.3)
    arrow(ax, (75, 44), (70, 37), rad=-0.12, lw=1.3)
    arrow(ax, (33, 24), (16, 17), rad=0.12, lw=1.3)
    ax.text(21.5, 19.6, f"{pi}*", fontsize=8.5, color=INK, fontweight="bold")
    for x0, x1 in [(22, 25), (44, 47), (66, 69)]:
        arrow(ax, (x0, 11.5), (x1, 11.5), lw=1.3)

    # clean closed-loop feedback along right margin and top
    ax.plot([88, 96.5], [11.5, 11.5], color=RED, lw=1.3)
    ax.plot([96.5, 96.5], [11.5, 61], color=RED, lw=1.3)
    ax.plot([96.5, 12], [61, 61], color=RED, lw=1.3)
    arrow(ax, (12, 61), (12, 56.4), lw=1.3, color=RED)
    ax.text(54, 62.9, f"CLOSED-LOOP FEEDBACK   {G_BULL}   next observation "
                      f"{ra} state s{G_PRIME},   reward r",
            ha="center", fontsize=8.0, color=RED, fontweight="bold")
    arrow(ax, (91, 17), (91, 30), lw=1.0, color=GREY, ls=(0, (3, 2)))
    ax.text(92.6, 24, "reward r used to\nupdate Q(s,a)", fontsize=6.8,
            color=GREY, va="center", ha="left", rotation=90)

    fig.savefig(FIG + "fig1_architecture.png")
    plt.close(fig)


# ================================================================= #
# FIG 2 - MDP state-transition diagram                               #
# ================================================================= #
def fig_mdp_graph():
    fig, ax = canvas(10.0, 5.7, (0, 112), (-7, 62),
                     "Figure 2   MDP State-Transition Structure of the "
                     "Patient Health Model")

    xs = [14, 34, 54, 74, 94]
    names = ["S0\nCritical", "S1\nSevere", "S2\nModerate", "S3\nMild",
             "S4\nStable"]
    cols = [LRED, LORANGE, "#fbf1d5", LGREEN, "#cfe6d8"]
    rx, ry = 7.0, 5.6
    for x, nm, c in zip(xs, names, cols):
        ax.add_patch(Ellipse((x, 33), 2 * rx, 2 * ry, facecolor=c,
                             edgecolor=INK, linewidth=1.1))
        ax.text(x, 33, nm, ha="center", va="center", fontsize=8.2,
                fontweight="bold", linespacing=1.3)

    # terminal states
    for x, nm, c in [(94, f"S5\nRecovered", "#bcdcc9"),
                     (14, f"S6\nAdverse", "#e8bdb9")]:
        ax.add_patch(Ellipse((x, 9), 2 * rx + 2.2, 2 * ry + 1.8,
                             facecolor="none", edgecolor=INK, linewidth=0.8))
        ax.add_patch(Ellipse((x, 9), 2 * rx, 2 * ry, facecolor=c,
                             edgecolor=INK, linewidth=1.1))
        ax.text(x, 9, nm, ha="center", va="center", fontsize=8.2,
                fontweight="bold", linespacing=1.3)

    # improvement / deterioration arcs
    for i in range(4):
        arrow(ax, (xs[i] + rx - 0.3, 35.6), (xs[i + 1] - rx + 0.3, 35.6),
              rad=-0.30, color=GREEN, lw=1.3)
        arrow(ax, (xs[i + 1] - rx + 0.3, 30.4), (xs[i] + rx - 0.3, 30.4),
              rad=-0.30, color=RED, lw=1.1, ls=(0, (4, 2)))

    # self-loops
    for x in xs:
        arrow(ax, (x - 3.0, 38.4), (x + 3.0, 38.4), rad=-2.1, color=GREY,
              lw=0.9)

    # to terminals
    arrow(ax, (94, 27.2), (94, 15.0), color=GREEN, lw=1.6)
    arrow(ax, (14, 27.2), (14, 15.0), color=RED, lw=1.6)
    arrow(ax, (69.5, 28.3), (20.5, 12.3), rad=0.13, color=RED, lw=0.9,
          ls=(0, (3, 2)))
    arrow(ax, (69.0, 28.6), (87.6, 12.6), rad=-0.30, color=GREEN, lw=1.0,
          ls=(0, (3, 2)))

    ax.text(94, 0.6, "absorbing", ha="center", fontsize=7, color=GREY,
            style="italic")
    ax.text(14, 0.6, "absorbing", ha="center", fontsize=7, color=GREY,
            style="italic")

    ax.text(54, 47.5, f"clinical improvement (recovery direction)",
            ha="center", fontsize=8.4, color=GREEN, fontweight="bold")
    ax.text(54, 21.5, f"deterioration (risk direction)", ha="center",
            fontsize=8.4, color=RED, fontweight="bold")
    ax.text(54, 58.0,
            f"A = {{ {a_[0]} Monitor,  {a_[1]} Medication,  "
            f"{a_[2]} Intensive Therapy,  {a_[3]} ICU Escalation,  "
            f"{a_[4]} Discharge }}",
            ha="center", fontsize=8.4, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#bbbbbb"))
    ax.text(54, 52.6, "grey self-loop = clinical condition unchanged during "
                      "the 8-hour decision epoch",
            ha="center", fontsize=7.4, color=GREY)
    ax.text(54, -4.2, f"R(s,a) = w{SUB[1]}{G_BULL}benefit  −  "
                     f"w{SUB[2]}{G_BULL}cost  −  w{SUB[3]}{G_BULL}"
                     f"adverse-risk  −  w{SUB[4]}{G_BULL}stay        "
                     f"terminal: +40 (recovery) / −60 (adverse)",
            ha="center", fontsize=7.6, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fafafa", ec="#cccccc"))

    fig.savefig(FIG + "fig2_mdp_state_diagram.png")
    plt.close(fig)


# ================================================================= #
# FIG 3 - Value-iteration flowchart                                  #
# ================================================================= #
def fig_flowchart_vi():
    fig, ax = canvas(6.8, 8.8, (0, 68), (0.5, 99.5),
                     "Figure 3   Flowchart — Value Iteration for the "
                     "Optimal Treatment Policy")

    cx = 31
    oval(ax, cx, 95.5, 26, 6.6, "START")
    box(ax, cx - 18, 84.0, 36, 7.6,
        f"Load MDP:   S,  A,  P(s{pr}|s,a),  R(s,a),  {g}", fc=LBLUE, fs=8)
    box(ax, cx - 18, 74.0, 36, 7.6,
        f"Initialise V(s) {la} 0  ∀ s {ein} S\n"
        f"set tolerance {th} = 10⁻⁸,   sweep k {la} 0",
        fc=LBLUE, fs=8)
    box(ax, cx - 18, 64.0, 36, 7.0,
        f"k {la} k + 1;   V_old {la} V;   {G_DELTA} {la} 0", fc=LGREY, fs=8)
    box(ax, cx - 21, 51.5, 42, 10.0,
        f"For every state s {ein} S apply the\n"
        f"Bellman optimality backup:\n"
        f"Q(s,a) = R(s,a) + {g} {sig}ₛ′ P(s{pr}|s,a) V_old(s{pr})",
        fc=LGREEN, fs=7.8)
    box(ax, cx - 18, 42.0, 36, 7.4,
        f"V(s) {la} maxₐ Q(s,a)\n"
        f"{G_DELTA} {la} max({G_DELTA}, |V(s) − V_old(s)|)",
        fc=LGREEN, fs=8)
    diamond(ax, cx, 31.0, 36, 13.5,
            f"Bellman residual\n{G_DELTA} < {th} ?", fs=8.2)
    box(ax, cx - 21, 13.5, 42, 8.4,
        f"Extract greedy policy\n"
        f"{pi}*(s) = arg maxₐ [ R(s,a) + {g}{sig} P(s{pr}|s,a) V*(s{pr}) ]",
        fc=LORANGE, fs=7.5)
    oval(ax, cx, 4.8, 32, 6.8, f"STOP  —  return V*, {pi}*", fc="#cfe6d8",
         fs=8.2)

    for y0, y1 in [(92.2, 91.7), (84.0, 81.7), (74.0, 71.1), (64.0, 61.6),
                   (51.5, 49.5), (42.0, 37.8)]:
        arrow(ax, (cx, y0), (cx, y1))
    arrow(ax, (cx, 24.2), (cx, 21.9))
    ax.text(cx + 3.2, 23.0, "Yes", fontsize=8.2, fontweight="bold")
    arrow(ax, (cx, 13.5), (cx, 8.3))

    ax.plot([cx + 18, 60], [31, 31], color=INK, lw=1.0)
    ax.plot([60, 60], [31, 67.5], color=INK, lw=1.0)
    arrow(ax, (60, 67.5), (cx + 18, 67.5), lw=1.0)
    ax.text(61.6, 49, "No  (next sweep)", fontsize=8, rotation=90,
            va="center", fontweight="bold")

    fig.savefig(FIG + "fig3_flowchart_value_iteration.png")
    plt.close(fig)


# ================================================================= #
# FIG 4 - End-to-end methodology pipeline                            #
# ================================================================= #
def fig_methodology():
    fig, ax = canvas(10.2, 5.2, (0, 100), (0, 52),
                     "Figure 4   Six-Phase Methodology of the Proposed System")

    ph = [
        ("PHASE 1\nProblem\nFormulation",
         f"Define S, A, P, R, {g}\nfrom clinical protocol", LBLUE),
        ("PHASE 2\nEnvironment\nConstruction",
         "Build simulator,\nvalidate P rows", LBLUE),
        ("PHASE 3\nPlanning\n(model-based)",
         f"Value + policy\niteration {ra} V*, {pi}*", LGREEN),
        ("PHASE 4\nLearning\n(model-free)",
         "Q-learning / SARSA\nfrom trajectories", LGREEN),
        ("PHASE 5\nEvaluation &\nBenchmarking",
         "Recovery rate, reward,\neffectiveness", LORANGE),
        ("PHASE 6\nDeployment &\nOversight",
         "Clinician-in-the-loop\nrecommendation UI", LRED),
    ]
    w, gap = 14.0, 2.4
    x = 2.0
    for title, sub, c in ph:
        box(ax, x, 26, w, 15, title, fc=c, fs=8.0, weight="bold")
        box(ax, x, 12, w, 11, sub, fc="#fafafa", fs=6.6)
        arrow(ax, (x + w / 2, 26), (x + w / 2, 23.3), lw=0.9, color=GREY)
        if x > 2.0:
            arrow(ax, (x - gap, 33.5), (x, 33.5), lw=1.4)
        x += w + gap

    ax.text(50, 46.5, "verification gate after every phase   "
                      "(unit tests · probability and value sanity checks "
                      "· clinical plausibility review)",
            ha="center", fontsize=7.8, color=GREY,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc"))
    ax.plot([91, 91], [12, 6], color=GREY, lw=1.0, ls=(0, (3, 2)))
    ax.plot([91, 9], [6, 6], color=GREY, lw=1.0, ls=(0, (3, 2)))
    arrow(ax, (9, 6), (9, 11.8), lw=1.0, color=GREY, ls=(0, (3, 2)))
    ax.text(50, 3.2, "refinement loop — reward re-weighting and "
                     "transition re-estimation from newly observed outcomes",
            ha="center", fontsize=7.4, color=GREY, style="italic")
    fig.savefig(FIG + "fig4_methodology_pipeline.png")
    plt.close(fig)


# ================================================================= #
# FIG 5 - Agent-environment interaction loop                         #
# ================================================================= #
def fig_agent_loop():
    fig, ax = canvas(8.4, 4.0, (0, 100), (0, 48),
                     "Figure 5   Agent\u2013Environment Interaction in the "
                     "Clinical MDP")

    box(ax, 10, 26, 32, 11,
        f"RL AGENT\n(treatment recommender)\n\n{pi}*(s) = arg max\u2090 Q*(s,a)",
        fc=LGREEN, fs=8.4)
    box(ax, 58, 26, 32, 11,
        f"ENVIRONMENT\n(patient + ward process)\n\nP(s{pr}|s,a),   R(s,a)",
        fc=LBLUE, fs=8.4)

    arrow(ax, (42, 34.2), (58, 34.2), lw=1.5)
    ax.text(50, 39.2, f"action a\u209c  (treatment)", ha="center",
            fontsize=7.8, fontweight="bold")
    arrow(ax, (58, 29.0), (42, 29.0), lw=1.5, color=BLUE)
    ax.text(50, 23.2, f"state s\u209c\u208a\u2081,  reward r\u209c\u208a\u2081",
            ha="center", fontsize=7.8, color=BLUE, fontweight="bold")

    box(ax, 4, 6, 92, 12,
        f"Reward composition\n"
        f"r = w{SUB[1]}\u00b7clinical benefit  \u2212  w{SUB[2]}\u00b7treatment cost  "
        f"\u2212  w{SUB[3]}\u00b7adverse-event risk  \u2212  "
        f"w{SUB[4]}\u00b7length-of-stay penalty\n"
        f"weights  w{SUB[1]}=1.0,  w{SUB[2]}=0.6,  w{SUB[3]}=0.8,  w{SUB[4]}=1.0"
        f"          terminal outcomes  recovery = +40,  adverse event = \u221260",
        fc="#fafafa", fs=7.8)
    arrow(ax, (78, 26), (74, 18.4), lw=0.9, color=GREY, ls=(0, (3, 2)))
    ax.text(50, 44.0, "one decision epoch = one 8-hour clinical review window",
            ha="center", fontsize=7.8, color=GREY, style="italic")
    fig.savefig(FIG + "fig5_agent_loop.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_mdp_graph(); fig_flowchart_vi()
    fig_methodology(); fig_agent_loop()
    print("diagrams written")
