#!/usr/bin/env python3
"""
Figure 4 — the coupling architecture is shared.

(a) Class-ordered global coupling matrix J (mechanistic blocks; cells whose 94% CrI includes
    zero are greyed, from couplings.csv).
(b) Transfer test: predict each country's three-drug co-resistance from the SINGLE global J
    (fields refit to that country's 15 single-drug rates, couplings held fixed) and compare to
    observed; the independence model from the same single-drug rates is the grey baseline.

Needs the raw ATLAS extract (per-country single-drug rates + observed co-resistance) via the
ATLAS_CSV env var, plus the shared fit.npz and couplings.csv in data_derived/.
Run:  ATLAS_CSV=/path/to/Escherichia_coli.csv python make_F4_transfer.py
"""
import os, sys, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from maxent_freq import load_binary_matrix, sufficient_statistics, enumerate_states  # noqa: E402

# ============================== CONFIG ==============================
HERE      = os.path.dirname(__file__)
CSV_PATH  = os.environ.get("ATLAS_CSV", os.path.join(HERE, "..", "Escherichia_coli.csv"))
FIT_NPZ   = os.path.join(HERE, "..", "data_derived", "fit.npz")
COUPLINGS = os.path.join(HERE, "..", "data_derived", "couplings.csv")
OUT_PNG   = os.path.join(HERE, "..", "figures_png", "F4_transfer.png")
COUNTRIES = ["United States", "China", "India", "Spain"]
COUNTRY_COL = {"United States": "#2b6cb0", "China": "#c0392b", "India": "#e67e22", "Spain": "#27ae60"}

CLASS_ORDER = ["Ampicillin","Ampicillin sulbactam","Amoxycillin clavulanate","Piperacillin tazobactam",
               "Cefepime","Ceftazidime","Ceftaroline","Aztreonam","Imipenem","Meropenem",
               "Ciprofloxacin","Levofloxacin","Amikacin","Gentamicin","Trimethoprim sulfa"]
SHORT = {"Ampicillin":"AMP","Ampicillin sulbactam":"AMS","Amoxycillin clavulanate":"AMC","Piperacillin tazobactam":"TZP",
         "Cefepime":"FEP","Ceftazidime":"CAZ","Ceftaroline":"CPT","Aztreonam":"ATM","Imipenem":"IPM","Meropenem":"MEM",
         "Ciprofloxacin":"CIP","Levofloxacin":"LVX","Amikacin":"AMK","Gentamicin":"GEN","Trimethoprim sulfa":"SXT"}
DPI = 150
# ===================================================================


def fit_fields_given_J(m_target, Jfix, S, SP, h0, lr=0.5, iters=8000, tol=1e-8):
    """Convex: match single-drug rates with couplings held fixed (robust GD from a warm start)."""
    h = h0.copy()
    for _ in range(iters):
        e = S @ h + SP @ Jfix; e -= e.max(); w = np.exp(e); p = w / w.sum()
        g = p @ S - m_target
        if np.abs(g).max() < tol:
            break
        h -= lr * g
    e = S @ h + SP @ Jfix; e -= e.max(); w = np.exp(e); p = w / w.sum()
    return p


def main():
    fit = np.load(FIT_NPZ, allow_pickle=True)
    h_glob, J_glob = fit["h"], fit["J"]
    drugs = list(fit["drugs"]); pairs = [tuple(x) for x in fit["pairs"]]; k = len(drugs)
    S, SP = enumerate_states(k, pairs)
    trip = list(itertools.combinations(range(k), 3))
    ST = np.column_stack([S[:, a]*S[:, b]*S[:, c] for a, b, c in trip]).astype(float)

    # uncertainty mask for panel (a)
    cb = pd.read_csv(COUPLINGS); pidx = {f"{drugs[i]} ~ {drugs[j]}": p for p, (i, j) in enumerate(pairs)}
    unc = np.zeros(len(pairs), bool)
    for _, r in cb.iterrows():
        a, b = r["pair"].split(" ~ "); p = pidx.get(r["pair"], pidx.get(f"{b} ~ {a}"))
        if p is not None and not bool(r["excludes_zero"]):
            unc[p] = True

    # data + per-country marginals/observed
    X = load_binary_matrix(CSV_PATH)[drugs]
    country = pd.read_csv(CSV_PATH, dtype=str).loc[X.index, "Country"].values

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.05))

    # ---- (a) global J blocks, uncertain greyed ----
    idx = {d: i for i, d in enumerate(drugs)}; order = [d for d in CLASS_ORDER if d in drugs]
    Jmat = np.full((k, k), np.nan); Umat = np.zeros((k, k), bool)
    for (i, j), v, u in zip(pairs, J_glob, unc):
        Jmat[i, j] = Jmat[j, i] = v; Umat[i, j] = Umat[j, i] = u
    perm = [idx[d] for d in order]; Jp = Jmat[np.ix_(perm, perm)]; Up = Umat[np.ix_(perm, perm)]
    np.fill_diagonal(Jp, np.nan); labs = [SHORT[d] for d in order]
    vmax = np.nanmax(np.abs(Jp))
    im = ax[0].imshow(np.where(Up, np.nan, Jp), cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax))
    for i in range(k):
        for j in range(k):
            if i != j and Up[i, j]:
                ax[0].add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor="#d9d9d9", edgecolor="none"))
    ax[0].set_xticks(range(k)); ax[0].set_xticklabels(labs, rotation=90, fontsize=8)
    ax[0].set_yticks(range(k)); ax[0].set_yticklabels(labs, fontsize=8)
    for b in [4, 8, 10, 12, 14]:
        ax[0].axhline(b-.5, color="k", lw=.8); ax[0].axvline(b-.5, color="k", lw=.8)
    fig.colorbar(im, ax=ax[0], fraction=.046, pad=.04).set_label("coupling J (grey = 94% CrI includes 0)")
    ax[0].set_title("(a) Coupling matrix J — mechanistic blocks (fit globally)", fontsize=10)

    # ---- (b) transfer ----
    obs_all, ind_all = [], []
    for c in COUNTRIES:
        Xc = X[country == c]; Xv = Xc.values
        m_c = Xc.values.mean(0)
        p_tr = fit_fields_given_J(m_c, J_glob, S, SP, h_glob.copy())
        p_ind = np.prod(S * m_c + (1 - S) * (1 - m_c), axis=1)
        obs_T = np.array([(Xv[:, a]*Xv[:, b]*Xv[:, c2]).mean() for a, b, c2 in trip])
        ax[1].scatter(obs_T, p_ind @ ST, s=5, alpha=.15, color="#b0b0b0", edgecolor="none")
        ax[1].scatter(obs_T, p_tr @ ST, s=9, alpha=.55, color=COUNTRY_COL[c], edgecolor="none", label=c)
        obs_all.append(obs_T)
    lim = max(np.concatenate(obs_all).max(), 1e-9) * 1.05
    ax[1].scatter([], [], s=5, color="#b0b0b0", label="independence (local rates only)")
    ax[1].plot([0, lim], [0, lim], "--", color="#e53e3e", lw=1)
    ax[1].set(xlim=(0, lim), ylim=(0, lim), xlabel="observed 3-drug co-resistance (per country)", ylabel="predicted")
    ax[1].set_aspect("equal"); ax[1].grid(alpha=.2); ax[1].legend(fontsize=8, loc="upper left", framealpha=.9)
    ax[1].set_title("(b) Transfer: global J + each country's single-drug rates", fontsize=10)

    # fig.suptitle("The coupling architecture is shared: one global J predicts every country's joint co-resistance",
                 fontsize=12, y=1.01)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
