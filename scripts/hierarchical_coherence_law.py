"""
Hierarchical Coherence Law — Final Clean Experiment
====================================================
Two validated findings:

FINDING 1: The Tight Staleness Bound
  The paper's staleness term (ζn/2) comes from the triangle inequality
  and is an upper bound. For the EXACT W2 staleness of the uniform-window
  estimator on a W2-Lipschitz path:
  
    W2(P̄_n, P*_t) ≤ (1/n) Σ_{j=0}^{n-1} W2(P*_{t-j}, P*_t)
  
  For a random-walk target (E[W2(P*_{t-j}, P*_t)] = ζ√j · √(2/π)):
  
    E[staleness] ≤ ζ · √(2/π) · (2/3) · √n   [tight, scales as √n not n]
  
  This gives a SQUARE-ROOT law for stochastic drift:
    n*(stochastic) = C_K · √2 / ζ  ~  1/ζ
    E_min(stochastic) = 2√(C_K·ζ/√2)  ~  ζ^{1/2}
  
  The cube-root law is the ADVERSARIAL bound (worst-case drift path).

FINDING 2: The Level-2 Cube-Root Law
  α*(κ) = (2κd/σ)^{2/3}
  is the optimal EMA coefficient when the drift rate ζ_t itself drifts
  with rate κ. This is the same cube-root structure as Level 1, applied
  to the drift estimator.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

rng = np.random.default_rng(0)

SIGMA = 1.0
D = 50
C_K = SIGMA * np.sqrt(2 / np.pi)         # 0.7979
C_ZETA = SIGMA * np.sqrt(2) / D          # 0.0283
C_S = np.sqrt(2 / np.pi) * (2 / 3)       # tight staleness constant = 0.5319

print(f"C_K = {C_K:.4f}  (finite-sample constant)")
print(f"C_s = {C_S:.4f}  (tight staleness constant, replaces 1/2)")
print(f"C_ζ = {C_ZETA:.4f}  (Level-2 constant)")
print()


# ── Theory ────────────────────────────────────────────────────────────────────

def E_paper(n, zeta, C_K=C_K):
    """Paper's bound: C_K·n^{-1/2} + ζ·n/2"""
    return C_K * n ** (-0.5) + zeta * n / 2

def E_tight(n, zeta, C_K=C_K, C_s=C_S):
    """Tight bound (stochastic drift): C_K·n^{-1/2} + ζ·C_s·n^{1/2}"""
    return C_K * n ** (-0.5) + zeta * C_s * n ** 0.5

def n_star_paper(zeta): return (C_K / zeta) ** (2 / 3)
def n_star_tight(zeta): return C_K * np.sqrt(2) / zeta  # = C_K / (zeta * C_s) with C_s=1/sqrt(2)

def E_min_paper(zeta): return 1.5 * C_K ** (2 / 3) * zeta ** (1 / 3)
def E_min_tight(zeta): return 2 * np.sqrt(C_K * zeta * C_S)

def alpha_star_L2(kappa):
    """Optimal EMA α for drift estimator: α* = (2κd/σ)^{2/3}"""
    return np.clip((2 * kappa * D / SIGMA) ** (2 / 3), 0.005, 0.95)


# ── Experiment 1: Staleness scaling measurement ────────────────────────────────

def exp1_staleness():
    print("=" * 65)
    print("EXP 1: Measuring tight staleness scaling")
    print("=" * 65)
    zeta = 0.01
    T = 50_000

    mu = np.zeros(T)
    for t in range(1, T):
        mu[t] = mu[t - 1] + zeta * rng.choice([-1, 1])
    X = mu + SIGMA * rng.standard_normal(T)

    ns = np.array([10, 20, 30, 50, 75, 100, 150, 200, 300, 500])
    staleness_emp = []
    for n in ns:
        s = [abs(mu[max(0, t - n) : t + 1].mean() - mu[t])
             for t in range(T // 2, T)]
        staleness_emp.append(np.mean(s))
    staleness_emp = np.array(staleness_emp)

    # Fit slope
    slope, intercept = np.polyfit(np.log(ns), np.log(staleness_emp), 1)
    const = np.exp(intercept)

    print(f"  Empirical staleness ~ {const:.4f} · n^{slope:.3f}")
    print(f"  Paper bound:   ζ·n/2    scales as n^1.000")
    print(f"  Tight bound:   ζ·C_s·√n scales as n^0.500")
    print()
    print(f"  {'n':>6} {'empirical':>12} {'paper_bound':>13} {'tight_bound':>13}")
    for i, n in enumerate(ns):
        paper = zeta * n / 2
        tight = zeta * C_S * np.sqrt(n)
        print(f"  {n:>6} {staleness_emp[i]:>12.5f} {paper:>13.5f} {tight:>13.5f}")

    # Full MAE curve
    ns_all = np.concatenate([np.arange(5, 50, 3), np.arange(50, 600, 15)])
    maes = []
    for n in ns_all:
        errs = [abs(X[max(0, t - n) : t + 1].mean() - mu[t])
                for t in range(T // 2, T)]
        maes.append(np.mean(errs))
    maes = np.array(maes)

    emp_opt_n = ns_all[np.argmin(maes)]
    emp_opt_mae = maes.min()
    print()
    print(f"  Empirical optimum:   n* = {emp_opt_n},  MAE = {emp_opt_mae:.4f}")
    print(f"  Paper cube-root:     n* = {n_star_paper(zeta):.1f}, E_min = {E_min_paper(zeta):.4f}")
    print(f"  Tight (stochastic):  n* = {n_star_tight(zeta):.1f}, E_min = {E_min_tight(zeta):.4f}")

    return ns, staleness_emp, slope, ns_all, maes, zeta


# ── Experiment 2: E_min scaling vs ζ ──────────────────────────────────────────

def exp2_emin_scaling():
    print()
    print("=" * 65)
    print("EXP 2: E_min scaling vs ζ — cube-root vs square-root")
    print("=" * 65)

    zetas = np.logspace(-2.5, -0.8, 10)
    T = 30_000
    SEEDS = 6
    emp_mins = []

    for zeta in zetas:
        # Wide sweep to find empirical minimum
        ns_sweep = np.concatenate([
            np.arange(5, 50, 5),
            np.arange(50, 400, 20),
            np.arange(400, int(min(4.0 / zeta, 3000)), 50),
        ])
        seed_maes_by_n = np.zeros(len(ns_sweep))
        for seed in range(SEEDS):
            rng2 = np.random.default_rng(seed * 31 + 7)
            mu = np.zeros(T)
            for t in range(1, T):
                mu[t] = mu[t - 1] + zeta * rng2.choice([-1, 1])
            X = mu + SIGMA * rng2.standard_normal(T)
            for i, n in enumerate(ns_sweep):
                errs = [abs(X[max(0, t - n) : t + 1].mean() - mu[t])
                        for t in range(T // 2, T)]
                seed_maes_by_n[i] += np.mean(errs)
        best_mae = (seed_maes_by_n / SEEDS).min()
        emp_mins.append(best_mae)
        print(f"  ζ={zeta:.4f}: E_min_emp={best_mae:.4f} | "
              f"paper={E_min_paper(zeta):.4f} | tight={E_min_tight(zeta):.4f}")

    emp_mins = np.array(emp_mins)
    # Fit slopes
    slope_emp, _ = np.polyfit(np.log(zetas), np.log(emp_mins), 1)
    slope_paper = 1 / 3
    slope_tight = 1 / 2
    print()
    print(f"  Empirical E_min ~ ζ^{slope_emp:.3f}")
    print(f"  Paper prediction: ~ ζ^{slope_paper:.3f}  (cube-root)")
    print(f"  Tight prediction: ~ ζ^{slope_tight:.3f}  (square-root)")

    return zetas, emp_mins


# ── Experiment 3: Level-2 cube-root law ───────────────────────────────────────

def exp3_level2():
    print()
    print("=" * 65)
    print("EXP 3: Level-2 cube-root: α*(κ) scaling")
    print("=" * 65)

    ZETA = 0.01
    T = 8_000
    WARMUP = T // 2
    SEEDS = 8
    alphas = np.geomspace(0.005, 0.6, 18)
    kappas = [5e-5, 2e-4, 8e-4, 3e-3]

    results = {}
    for kappa in kappas:
        alpha_th = alpha_star_L2(kappa)
        mae_by_alpha = []
        for alpha in alphas:
            sm = []
            for seed in range(SEEDS):
                rng2 = np.random.default_rng(seed * 79 + 13)
                zp = np.zeros(T); zp[0] = ZETA
                for t in range(1, T):
                    zp[t] = np.clip(zp[t - 1] + kappa * 0.5 * rng2.choice([-1, 1]),
                                    1e-4, 0.5)
                mu = np.zeros(T)
                for t in range(1, T):
                    mu[t] = mu[t - 1] + zp[t - 1] * rng2.choice([-1, 1])
                X = mu + SIGMA * rng2.standard_normal(T)
                zh = np.zeros(T); zh[: 2 * D] = ZETA
                for t in range(2 * D, T):
                    b1 = X[t - D : t].mean(); b2 = X[t - 2 * D : t - D].mean()
                    zh[t] = alpha * abs(b1 - b2) / D + (1 - alpha) * zh[t - 1]
                # Use tight n* (larger window = better for stochastic drift)
                ns = np.clip((C_K * np.sqrt(2) / np.maximum(zh, 1e-9)).astype(int),
                             10, 2000)
                errs = [abs(X[max(0, t - ns[t]) : t + 1].mean() - mu[t])
                        for t in range(WARMUP, T)]
                sm.append(np.mean(errs))
            mae_by_alpha.append(np.mean(sm))

        mae_arr = np.array(mae_by_alpha)
        best_idx = np.argmin(mae_arr)
        alpha_emp = alphas[best_idx]
        results[kappa] = (alphas, mae_arr, alpha_th, alpha_emp)
        print(f"  κ={kappa:.1e} | α*(theory)={alpha_th:.4f} | "
              f"α*(empirical)={alpha_emp:.4f} | ratio={alpha_emp/alpha_th:.2f}")

    # Fit scaling exponent
    kappas_arr = np.array(kappas)
    alpha_emps = np.array([results[k][3] for k in kappas])
    slope, _ = np.polyfit(np.log(kappas_arr), np.log(alpha_emps), 1)
    print(f"\n  Empirical scaling: α* ~ κ^{slope:.3f}  (theory: κ^{2/3:.3f})")
    return results, kappas_arr, alpha_emps


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(e1, e2, e3):
    ns_s, stale_emp, slope_s, ns_all, maes, zeta = e1
    zetas, emp_mins = e2
    res3, kappas_arr, alpha_emps = e3

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#080808")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.38,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    ACC = ["#00e5ff", "#ff6b6b", "#a8ff78", "#ffd166", "#c77dff"]
    GC = "#1a1a1a"; TC = "#aaaaaa"

    def sax(ax, title, xl, yl):
        ax.set_facecolor("#101010")
        for sp in ax.spines.values(): sp.set_color("#2a2a2a")
        ax.tick_params(colors=TC, labelsize=8)
        ax.xaxis.label.set_color(TC); ax.yaxis.label.set_color(TC)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_title(title, color="white", fontsize=9.5, pad=6, fontweight="bold")
        ax.grid(True, color=GC, linewidth=0.6)

    # A: Staleness scaling
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(ns_s, stale_emp, "o-", color=ACC[0], lw=2, ms=6, label="Empirical staleness")
    ax.loglog(ns_s, zeta * ns_s / 2, "--", color=ACC[1], lw=1.5,
              label=r"Paper bound: $\zeta n/2$  ($\sim n^1$)")
    ax.loglog(ns_s, zeta * C_S * np.sqrt(ns_s), "-.", color=ACC[2], lw=1.8,
              label=r"Tight: $\zeta C_s \sqrt{n}$  ($\sim n^{1/2}$)")
    ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
    sax(ax, "A — Staleness Scaling\n(paper bound vs tight)",
        "Window $n$", r"$E[W_2(\bar{P}_n, P^*_t)]$")

    # B: Full MAE curve with both theory predictions
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ns_all, maes, color=ACC[0], lw=2, label="Empirical MAE")
    ax.plot(ns_all, E_paper(ns_all, zeta), "--", color=ACC[1], lw=1.8,
            label=r"Paper: $C_K n^{-1/2} + \zeta n/2$")
    ax.plot(ns_all, E_tight(ns_all, zeta), "-.", color=ACC[2], lw=1.8,
            label=r"Tight: $C_K n^{-1/2} + \zeta C_s n^{1/2}$")
    ax.axvline(n_star_paper(zeta), color=ACC[1], lw=1, ls=":", alpha=0.7,
               label=f"$n^*$ paper={n_star_paper(zeta):.0f}")
    ax.axvline(n_star_tight(zeta), color=ACC[2], lw=1, ls=":", alpha=0.7,
               label=f"$n^*$ tight={n_star_tight(zeta):.0f}")
    ax.set_xlim(0, 600); ax.set_ylim(0, 0.5)
    ax.legend(fontsize=6.5, facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
    sax(ax, f"B — Full MAE Curve ($\\zeta$={zeta})\nU-curve: paper vs tight theory",
        "Window $n$", "Tail MAE")

    # C: E_min vs zeta scaling
    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(zetas, emp_mins, "o-", color=ACC[0], lw=2, ms=6, label="Empirical $E_{\\min}$")
    ax.loglog(zetas, E_min_paper(zetas), "--", color=ACC[1], lw=1.8,
              label=r"Paper: $\propto \zeta^{1/3}$")
    ax.loglog(zetas, E_min_tight(zetas), "-.", color=ACC[2], lw=1.8,
              label=r"Tight: $\propto \zeta^{1/2}$")
    # Reference lines
    zr = zetas[[0, -1]]
    for exp, col, lbl in [(1/3, ACC[1], ""), (1/2, ACC[2], ""), (0.42, "white", "emp")]:
        pass
    ax.legend(fontsize=7.5, facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
    sax(ax, r"C — $E_{\min}$ vs $\zeta$: Which law?",
        r"Drift rate $\zeta$", r"Minimum achievable MAE")

    # D: Level-2 MAE vs alpha for each kappa
    ax = fig.add_subplot(gs[1, 0:2])
    colors_k = [ACC[0], ACC[1], ACC[2], ACC[3], ACC[4]]
    for i, kappa in enumerate(sorted(res3.keys())):
        alphas, mae_arr, alpha_th, alpha_emp = res3[kappa]
        col = colors_k[i]
        ax.plot(alphas, mae_arr, "-", color=col, lw=1.8,
                label=f"κ={kappa:.0e}  (α*={alpha_emp:.3f}, theory={alpha_th:.3f})")
        ax.axvline(alpha_th, color=col, lw=1.0, ls="--", alpha=0.6)
        ax.axvline(alpha_emp, color=col, lw=0.7, ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.legend(fontsize=7.5, facecolor="#1a1a1a", labelcolor="white",
              edgecolor="#444", loc="upper left", ncol=2)
    sax(ax, "D — Level-2 MAE vs α for each κ\n"
        "(dashed=theory α*, dotted=empirical best)",
        r"EMA coefficient $\alpha$", "Tail MAE")

    # E: alpha* scaling
    ax = fig.add_subplot(gs[1, 2])
    kth = np.logspace(-5, -2, 100)
    ax.loglog(kth, alpha_star_L2(kth), "--", color=ACC[1], lw=2,
              label=r"Theory: $\alpha^* = (2\kappa d/\sigma)^{2/3}$")
    ax.loglog(kappas_arr, alpha_emps, "o", color=ACC[0], ms=8, zorder=5,
              label="Empirical $\\alpha^*$")
    # Fit line
    slope_fit, ic = np.polyfit(np.log(kappas_arr), np.log(alpha_emps), 1)
    ax.loglog(kappas_arr, np.exp(ic) * kappas_arr ** slope_fit,
              "-", color=ACC[0], lw=1.5, alpha=0.6,
              label=f"Empirical fit: $\\sim \\kappa^{{{slope_fit:.2f}}}$")
    ax.legend(fontsize=7.5, facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
    sax(ax, r"E — Level-2 Cube-Root: $\alpha^*(\kappa)$",
        r"Drift-of-drift rate $\kappa$", r"Optimal EMA $\alpha^*$")

    fig.suptitle(
        "Hierarchical Coherence Law  —  Tight Staleness & Level-2 Cube-Root",
        color="white", fontsize=13, fontweight="bold", y=0.97)

    path = "hierarchical_law_final.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved: {path}")
    plt.close()


if __name__ == "__main__":
    e1 = exp1_staleness()
    e2 = exp2_emin_scaling()
    e3 = exp3_level2()
    make_figure(e1, e2, e3)