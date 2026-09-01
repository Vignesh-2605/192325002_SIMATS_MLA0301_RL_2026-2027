"""Result figures generated from the actual experiment outputs."""
import json
import numpy as np
import pandas as pd
from figstyle import *
from patient_mdp import STATE_LABEL, ACTION_LABEL, TERMINAL, S, A

FIG, RES = "figures/", "results/"
g, pi, la = G_GAMMA, G_PI, G_LARR
summary = json.load(open(RES + "summary.json"))
NT = [0, 1, 2, 3, 4]
NTL = [STATE_LABEL[s] for s in NT]


def mov(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


# ================================================================= #
# FIG 6 - Value-iteration convergence                                #
# ================================================================= #
def fig_convergence():
    delta = np.load(RES + "vi_history_delta.npy")
    Vh = np.load(RES + "vi_history_V.npy")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ax = axes[0]
    ax.semilogy(np.arange(1, len(delta) + 1), delta, color=BLUE, lw=1.6,
                marker="o", ms=2.6, mfc="white", mew=0.7)
    ax.axhline(1e-8, color=RED, lw=1.1, ls="--")
    ax.text(len(delta) * 0.52, 2.2e-8, f"tolerance {G_THETA} = 10⁻⁸",
            color=RED, fontsize=8)
    ax.set_xlabel("Value-iteration sweep  k")
    ax.set_ylabel(f"Bellman residual   maxₛ |Vₖ(s) − Vₖ₋₁(s)|")
    ax.set_title(f"(a)  Geometric convergence  ({g} = 0.95, "
                 f"{len(delta)} sweeps)")

    ax = axes[1]
    cols = [RED, ORANGE, "#b8952f", GREEN, BLUE]
    for i, s in enumerate(NT):
        ax.plot(np.arange(1, Vh.shape[0] + 1), Vh[:, s], lw=1.7,
                color=cols[i], label=f"V(s) : {STATE_LABEL[s]}")
    ax.set_xlabel("Value-iteration sweep  k")
    ax.set_ylabel("State value  V(s)")
    ax.set_title("(b)  Convergence of the state-value function")
    ax.legend(loc="lower right", ncol=1)
    ax.set_xlim(0, 60)

    fig.suptitle("Figure 6   Convergence of Value Iteration on the Clinical "
                 "MDP", fontsize=11.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG + "fig6_vi_convergence.png")
    plt.close(fig)


# ================================================================= #
# FIG 7 - Q*(s,a) heat map with optimal policy                       #
# ================================================================= #
def fig_qheatmap():
    Q = np.load(RES + "Q_star.npy")[NT, :]
    piv = np.load(RES + "pi_star.npy")

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    im = ax.imshow(Q, cmap="RdYlGn", aspect="auto", vmin=-60, vmax=50)
    ax.set_xticks(range(A)); ax.set_xticklabels(ACTION_LABEL)
    ax.set_yticks(range(len(NT))); ax.set_yticklabels(NTL)
    ax.set_xlabel("Treatment action  a"); ax.set_ylabel("Clinical state  s")
    ax.grid(False)
    for i, s in enumerate(NT):
        for j in range(A):
            best = (j == piv[s])
            ax.text(j, i, f"{Q[i, j]:.1f}", ha="center", va="center",
                    fontsize=8.4, fontweight="bold" if best else "normal",
                    color="black")
            if best:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.90,
                                           fill=False, edgecolor="black",
                                           lw=2.2))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Q*(s, a)", fontsize=8.5)
    ax.set_title("Figure 7   Optimal Action-Value Function Q*(s,a);  "
                 "boxed cell = recommended treatment " + f"{pi}*(s)",
                 fontsize=10.5, pad=9)
    fig.tight_layout()
    fig.savefig(FIG + "fig7_q_heatmap.png")
    plt.close(fig)


# ================================================================= #
# FIG 8 - Model-free learning curves                                 #
# ================================================================= #
def fig_learning():
    rq = np.load(RES + "ql_returns.npy")
    rs = np.load(RES + "sarsa_returns.npy")
    ep = np.load(RES + "ql_eps.npy")
    td = np.load(RES + "ql_td.npy")
    x = np.arange(1, len(rq) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))

    ax = axes[0]
    ax.plot(x, mov(rq, 400), color=BLUE, lw=1.6, label="Q-learning")
    ax.plot(x, mov(rs, 400), color=ORANGE, lw=1.6, label="SARSA")
    ax.axhline(summary["comparison"]["RL Optimal Policy (Value Iteration)"]
               ["Mean Cumulative Reward"], color=GREEN, ls="--", lw=1.4,
               label="DP optimum (value iteration)")
    ax.set_xlabel("Training episode (simulated patient)")
    ax.set_ylabel("Cumulative reward per episode\n(400-episode moving mean)")
    ax.set_title("(a)  Learning curves")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(x, mov(td, 100), color="#c9c0dd", lw=0.8)
    ax.plot(x, mov(td, 1000), color=PURPLE, lw=2.0,
            label="1 000-episode moving mean")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean absolute TD error  |δ| per episode")
    ax.set_title("(b)  Temporal-difference error decay")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(x, ep, color=RED, lw=1.7)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Exploration rate  {G_EPS}")
    ax.set_title("(c)  " + f"{G_EPS}-greedy exploration schedule")
    ax.set_ylim(0, 1.02)

    fig.suptitle("Figure 8   Model-Free Learning Behaviour "
                 "(30 000 simulated patient episodes)",
                 fontsize=11.5, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(FIG + "fig8_learning_curves.png")
    plt.close(fig)


# ================================================================= #
# FIG 9 - Policy benchmarking                                        #
# ================================================================= #
def fig_comparison():
    df = pd.read_csv(RES + "policy_comparison.csv", index_col=0)
    order = ["RL Optimal Policy (Value Iteration)",
             "RL Learned Policy (Q-Learning)", "Fixed Clinical Protocol",
             "Uniform Medication", "Conservative Monitoring",
             "Aggressive Escalation", "Random Policy"]
    df = df.loc[order]
    short = ["RL optimal\n(value iter.)", "RL learned\n(Q-learning)",
             "Fixed clinical\nprotocol", "Uniform\nmedication",
             "Conservative\nmonitoring", "Aggressive\nescalation",
             "Random\npolicy"]
    cols = [GREEN, "#5a9e78", BLUE, PURPLE, ORANGE, "#b06a2f", GREY]

    panels = [
        ("Recovery Rate (%)", "Recovery rate  (% of patients)",
         "(a)  Recovery rate", None),
        ("Mean Cumulative Reward", "Mean cumulative reward per patient",
         "(b)  Cumulative reward", 0),
        ("Treatment Effectiveness (%)",
         "Interventions producing improvement (%)",
         "(c)  Treatment effectiveness", None),
        ("Adverse Outcome Rate (%)", "Adverse outcome rate  (%)",
         "(d)  Adverse outcome rate", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))
    for ax, (col, ylab, title, zero) in zip(axes.ravel(), panels):
        v = df[col].to_numpy()
        b = ax.bar(range(len(v)), v, color=cols, edgecolor=INK, linewidth=0.7,
                   width=0.68)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(short, fontsize=6.9)
        ax.set_ylabel(ylab, fontsize=8.6)
        ax.set_title(title, fontsize=9.8)
        if zero is not None:
            ax.axhline(0, color=INK, lw=0.9)
        pad = (max(v) - min(min(v), 0)) * 0.04
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2,
                    val + (pad if val >= 0 else -pad * 2.4),
                    f"{val:.1f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=7.6,
                    fontweight="bold")
        ax.margins(y=0.16)
    fig.suptitle("Figure 9   Benchmarking the Learned Policy Against "
                 "Conventional Treatment Strategies  (5 000 simulated "
                 "patients each)", fontsize=11.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG + "fig9_policy_comparison.png")
    plt.close(fig)


# ================================================================= #
# FIG 10 - State-occupancy evolution                                 #
# ================================================================= #
def fig_occupancy():
    m_rl = np.load(RES + "occupancy_RL.npy")
    m_fx = np.load(RES + "occupancy_Fixed.npy")
    cols = [RED, ORANGE, "#c9b04a", "#7fae8e", BLUE, GREEN, "#7a1f1a"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=True)
    for ax, m, t in [(axes[0], m_rl, "(a)  RL optimal policy"),
                     (axes[1], m_fx, "(b)  Fixed clinical protocol")]:
        t_ax = np.arange(m.shape[0]) / 3.0
        ax.stackplot(t_ax, m.T * 100, colors=cols, labels=STATE_LABEL,
                     edgecolor="white", linewidth=0.35)
        ax.set_xlabel("Time since admission  (days)")
        ax.set_title(t, fontsize=10)
        ax.set_xlim(0, t_ax[-1]); ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.35)
    axes[0].set_ylabel("Share of the patient cohort  (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                   fontsize=8, title="Clinical state", title_fontsize=8.5)
    fig.suptitle("Figure 10   Evolution of Cohort Health States Over Time "
                 "(4 000 simulated patients)",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG + "fig10_state_occupancy.png")
    plt.close(fig)


# ================================================================= #
# FIG 11 - Treatment effectiveness per state                         #
# ================================================================= #
def fig_effectiveness():
    df = pd.read_csv(RES + "treatment_effectiveness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    ax = axes[0]
    y = np.arange(len(df))[::-1]
    imp = df["P(improve)"] * 100
    sam = df["P(unchanged)"] * 100
    wor = df["P(deteriorate)"] * 100
    ax.barh(y, imp, color=GREEN, edgecolor=INK, lw=0.6, label="improves")
    ax.barh(y, sam, left=imp, color="#d8d8d8", edgecolor=INK, lw=0.6,
            label="unchanged")
    ax.barh(y, wor, left=imp + sam, color=RED, edgecolor=INK, lw=0.6,
            label="deteriorates")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.State}\n{la} {r._2}" for r in
                        df.itertuples()], fontsize=7.8)
    ax.set_xlabel("One-step outcome probability under " + f"{pi}*(s)  (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.8)
    ax.set_title("(a)  Per-state treatment effectiveness", fontsize=10)
    for i, v in zip(y, imp):
        ax.text(v / 2, i, f"{v:.0f}%", ha="center", va="center",
                fontsize=7.8, color="white", fontweight="bold")

    ax = axes[1]
    adv = df["Advantage over 2nd best"]
    b = ax.bar(range(len(df)), adv, color=BLUE, edgecolor=INK, lw=0.7,
               width=0.6)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["State"], fontsize=8)
    ax.set_ylabel("Q*(s, " + f"{pi}*) − Q*(s, 2nd best)")
    ax.set_xlabel("Clinical state")
    ax.set_title("(b)  Decision margin of the recommended action",
                 fontsize=10)
    for r, v in zip(b, adv):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.25, f"{v:.2f}",
                ha="center", fontsize=7.8, fontweight="bold")
    ax.margins(y=0.18)

    fig.suptitle("Figure 11   Effectiveness and Confidence of the "
                 "Recommended Treatment in Each Clinical State",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG + "fig11_treatment_effectiveness.png")
    plt.close(fig)


# ================================================================= #
# FIG 12 - Discount-factor sensitivity                               #
# ================================================================= #
def fig_gamma():
    df = pd.read_csv(RES + "gamma_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5))

    ax = axes[0]
    ax.plot(df["gamma"], df["V*(Critical)"], marker="o", color=RED, lw=1.7,
            ms=5, mfc="white", label="V*(Critical)")
    ax.plot(df["gamma"], df["V*(Stable)"], marker="s", color=GREEN, lw=1.7,
            ms=5, mfc="white", label="V*(Stable)")
    ax.set_xlabel(f"Discount factor  {g}")
    ax.set_ylabel("Optimal state value  V*(s)")
    ax.set_title("(a)  Sensitivity of the value function", fontsize=10)
    ax.legend()

    ax = axes[1]
    mat = np.zeros((len(df), 5), dtype=int)
    for i, r in df.iterrows():
        for j, st in enumerate(NTL):
            mat[i, j] = ACTION_LABEL.index(r[st])
    cmap = matplotlib.colors.ListedColormap(
        ["#dfe6ec", "#b8d4c3", "#f0d9b5", "#e8b7b3", "#c7bcd9"])
    ax.imshow(mat.T, cmap=cmap, aspect="auto", vmin=-0.5, vmax=4.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{v:.2f}" for v in df["gamma"]])
    ax.set_yticks(range(5)); ax.set_yticklabels(NTL, fontsize=8.5)
    ax.set_xlabel(f"Discount factor  {g}")
    ax.grid(False)
    for i in range(len(df)):
        for j in range(5):
            ax.text(i, j, ACTION_LABEL[mat[i, j]].replace(" Escalation", ""),
                    ha="center", va="center", fontsize=7.0)
    ax.set_title(f"(b)  Recommended treatment {pi}*(s) versus {g}",
                 fontsize=10)

    fig.suptitle("Figure 12   Effect of the Discount Factor on the Learned "
                 "Clinical Policy", fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG + "fig12_gamma_sensitivity.png")
    plt.close(fig)


# ================================================================= #
# FIG 13 - Robustness with 95% confidence intervals                  #
# ================================================================= #
def fig_robustness():
    rb = summary["robustness"]
    names = list(rb.keys())
    short = ["RL optimal policy", "Fixed clinical protocol",
             "Uniform medication"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))
    specs = [("recovery", "Recovery rate (%)", "(a)  Recovery rate"),
             ("reward", "Mean cumulative reward", "(b)  Cumulative reward"),
             ("effectiveness", "Treatment effectiveness (%)",
              "(c)  Treatment effectiveness")]
    cols = [GREEN, BLUE, PURPLE]
    for ax, (k, ylab, t) in zip(axes, specs):
        mu = [rb[n][f"{k}_mean"] for n in names]
        ci = [rb[n][f"{k}_ci95"] for n in names]
        ax.bar(range(3), mu, yerr=ci, capsize=5, color=cols, edgecolor=INK,
               lw=0.7, width=0.6, error_kw=dict(lw=1.1, ecolor=INK))
        ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=7.2,
                                                    rotation=12, ha="right")
        ax.set_ylabel(ylab, fontsize=8.6)
        ax.set_title(t, fontsize=9.8)
        for i, (m, c) in enumerate(zip(mu, ci)):
            ax.text(i, m + c + (max(mu) * 0.03), f"{m:.1f}±{c:.2f}",
                    ha="center", fontsize=7.4, fontweight="bold")
        ax.margins(y=0.2)
    fig.suptitle("Figure 13   Repeated-Seed Robustness  "
                 "(10 independent runs × 2 000 patients, 95% CI)",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG + "fig13_robustness.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence(); fig_qheatmap(); fig_learning(); fig_comparison()
    fig_occupancy(); fig_effectiveness(); fig_gamma(); fig_robustness()
    print("result figures written")
