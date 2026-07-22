#!/usr/bin/env python3
"""
=============================================================================================
PAIRWISE MAXIMUM-ENTROPY (ISING) MODEL OF BINARY RESISTANCE VECTORS  —  Bayesian / PyMC
=============================================================================================

THE QUESTION
------------
Each isolate gives a binary vector x = (x_1, ..., x_k), one entry per drug:
    x_d = 1  if non-susceptible (I or R),  0 if susceptible (S).
The population is a probability distribution over the 2^k possible vectors.

We ask: is that distribution fully explained by (a) how often each drug is resistant, and
(b) how often each PAIR of drugs is jointly resistant?  If yes, there is no "specific
higher-order epistasis" — no irreducible 3-drug (or 4-drug) rule beyond what pairs imply.

THE MODEL
---------
The maximum-entropy distribution that matches the observed single-drug rates and pairwise
rates — and assumes nothing else — is the pairwise (Ising) model:

    P(x)  =  exp( sum_d h_d x_d  +  sum_{d<d'} J_dd' x_d x_d' )  /  Z

    Z     =  sum over all 2^k vectors of the same exponential      (partition function)

    h_d   = "field"    : intrinsic propensity of drug d to be resistant
    J_dd' = "coupling" : co-resistance of d and d' BEYOND what their individual rates imply
                         (J > 0 co-occur more than expected; J < 0 less than expected)

By construction the model contains NO 3-way or higher interaction term. So if it reproduces
the observed 3-drug frequencies, pairs are sufficient. That is the whole logic.

WHY THIS RUNS FAST DESPITE N = 45,000
-------------------------------------
The log-likelihood of the whole dataset is

    log L  =  sum_i log P(x_i)
           =  N * [ sum_d h_d * m_d  +  sum_{d<d'} J_dd' * C_dd'  -  log Z ]

where  m_d   = observed fraction of isolates non-susceptible to d          (a length-k vector)
       C_dd' = observed fraction non-susceptible to BOTH d and d'          (one per pair)

The data enters ONLY through m and C — these are "sufficient statistics". So we compute them
once, and the sampler never touches the 45,000 rows again. Runtime depends on k, not N.

REQUIREMENTS
------------
    pip install pymc arviz numpy pandas
Tested against PyMC 5.x.  Start with N_DRUGS = 10 to check it runs, then raise it.

=============================================================================================
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# =============================================================================================
# 1. CONFIGURATION
# =============================================================================================

SCRIPT_DIR = Path(__file__).parent

CSV_PATH = SCRIPT_DIR / "Escherichia coli_pca.csv"   # <-- Now points to the script's folder[cite: 1]
OUT_DIR = SCRIPT_DIR / "maxent_out"                  # <-- Keeps your output folder there too[cite: 1]

# Number of drugs to model. The model enumerates 2^k states explicitly, so cost grows as 2^k:
#   k=10 ->    1,024 states  (fast, ~1-2 min)
#   k=12 ->    4,096 states  (~5 min)
#   k=15 ->   32,768 states  (~15-40 min)
#   k=18 ->  262,144 states  (slow + memory-hungry; consider this the ceiling)
# Drugs are chosen by highest variance (most informative); degenerate drugs carry no coupling.
N_DRUGS = 15

MIN_COVERAGE = 0.95   # keep a drug only if its _I column is >=95% populated
MIN_FREQ = 0.02       # drop drugs that are <2% or >98% resistant (no information, h diverges)
BINARISATION = "NS"   # "NS": non-susceptible (I or R) = 1   |   "R": only R = 1

# Priors. These are weakly informative: they regularise without dictating the answer.
# NOTE couplings can genuinely be large (|J| ~ 6-8 for drugs sharing one mechanism), so do NOT
# tighten SIGMA_J much below ~2.5 or you will shrink real couplings toward zero.
SIGMA_H = 5.0
SIGMA_J = 2.5

# Higher-order epistasis test (Section: check_all_orders).
# MAX_ORDER = highest interaction order to test (None = all the way up to k). Cost of order o is
# C(k, o) tuples, which peaks near o = k/2, so testing every order costs ~2^k tuple evaluations total.
# MAX_TUPLES_PER_ORDER caps how many tuples are checked per order: if C(k, o) exceeds it, a random
# sample is drawn (seeded), which keeps memory/time bounded while still estimating the order's fit.
MAX_ORDER = None
MAX_TUPLES_PER_ORDER = 10000
TUPLE_CHUNK = 512          # tuples processed at a time (memory control; each chunk is 2^k x chunk)

# Sampler settings
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.9
SEED = 42


# =============================================================================================
# 2. LOAD AND BINARISE THE DATA
# =============================================================================================

def load_binary_matrix(csv_path, binarisation=BINARISATION,
                       min_coverage=MIN_COVERAGE, min_freq=MIN_FREQ):
    """
    Read the ATLAS-style CSV and return a DataFrame of 0/1 resistance calls (one column per drug).

    Each drug has two columns: the raw MIC (e.g. "Meropenem") and the laboratory's interpretation
    ("Meropenem_I", with values Susceptible / Intermediate / Resistant). We use the interpretation
    column: the lab has already applied the clinical breakpoint, which conveniently sidesteps all
    the interval-censoring in the raw MIC strings ("<=0.5", ">32").
    """
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    def interpret(value):
        """Susceptible/Intermediate/Resistant  ->  0 / 1 / NaN."""
        if pd.isna(value):
            return np.nan
        v = str(value).strip().lower()
        if binarisation == "NS":                       # non-susceptible = I or R
            if v.startswith("resist") or v.startswith("interm"):
                return 1.0
            if v.startswith("suscept"):
                return 0.0
        else:                                          # strict: only R counts
            if v.startswith("resist"):
                return 1.0
            if v.startswith("interm") or v.startswith("suscept"):
                return 0.0
        return np.nan

    # Step 1: find well-populated interpretation columns.
    interp_cols = [c for c in df.columns
                   if c.endswith("_I") and df[c].notna().mean() >= min_coverage]

    # Step 2: binarise, and name columns after the drug (strip the "_I").
    X = pd.DataFrame({c[:-2]: df[c].map(interpret) for c in interp_cols})

    # Step 3: complete cases only. ATLAS missingness is structured (drugs are run in panels),
    # so this drops very few rows once we have restricted to dense columns.
    X = X.dropna()

    # Step 4: drop degenerate drugs (nearly always resistant or nearly always susceptible).
    # They carry no co-resistance information and their field h would run off to +/- infinity.
    keep = [c for c in X.columns if min_freq < X[c].mean() < 1 - min_freq]
    X = X[keep]

    print(f"Loaded {len(X):,} complete-case isolates x {X.shape[1]} non-degenerate drugs")
    return X


def select_drugs(X, n_drugs):
    """Keep the n most variable drugs. Variance of a Bernoulli is p(1-p), maximal near p=0.5."""
    variance = X.mean() * (1 - X.mean())
    chosen = list(variance.sort_values(ascending=False).index[:n_drugs])
    print(f"Modelling {len(chosen)} drugs: {', '.join(chosen)}")
    return X[chosen]


# =============================================================================================
# 3. SUFFICIENT STATISTICS AND STATE ENUMERATION
# =============================================================================================

def sufficient_statistics(X):
    """
    Compress the dataset to what the likelihood actually needs.

    Returns
        m      : (k,)       fraction of isolates resistant to each drug
        C_vec  : (n_pairs,) fraction resistant to BOTH drugs, for each pair (i<j)
        pairs  : list of (i, j) index tuples, in the same order as C_vec
    """
    Xv = X.values
    k = Xv.shape[1]
    m = Xv.mean(axis=0)

    pairs = list(itertools.combinations(range(k), 2))
    C_vec = np.array([(Xv[:, i] * Xv[:, j]).mean() for i, j in pairs])

    return m, C_vec, pairs


def enumerate_states(k, pairs):
    """
    Build every possible resistance vector and its pairwise products.

    Returns
        S  : (2^k, k)        every binary vector, one per row
        SP : (2^k, n_pairs)  SP[s, p] = 1 iff state s is resistant to BOTH drugs of pair p

    These two matrices turn the energy of every state into one matrix product:
        energy = S @ h + SP @ J_vec
    which is exactly  sum_d h_d x_d + sum_{d<d'} J_dd' x_d x_d'  evaluated for all states at once.
    (This is what the old script's einsum was doing, written more plainly.)
    """
    S = np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.float64)
    SP = np.column_stack([S[:, i] * S[:, j] for i, j in pairs])
    print(f"Enumerated {S.shape[0]:,} states x {SP.shape[1]} pairs")
    return S, SP


def triple_design(S, k):
    """
    (2^k, n_triples) indicator matrix: 1 iff the state is resistant to all three drugs.
    Used only for the epistasis test AFTER fitting — the model itself never sees triples.
    float32 keeps memory sane at larger k.
    """
    triples = list(itertools.combinations(range(k), 3))
    ST = np.column_stack([S[:, i] * S[:, j] * S[:, l] for i, j, l in triples]).astype(np.float32)
    return ST, triples


# =============================================================================================
# 4. THE PYMC MODEL
# =============================================================================================

def build_model(m, C_vec, S, SP, n_obs):
    """
    The whole model in one place.

    Parameters:  h (k,) fields, J (n_pairs,) couplings.
    Likelihood:  the exact Ising log-likelihood, added with pm.Potential because it is not one
                 of PyMC's built-in distributions — we are writing it out by hand.
    """
    k = S.shape[1]
    n_pairs = SP.shape[1]

    with pm.Model() as model:

        # ---- Priors -------------------------------------------------------------------------
        # Weakly informative. The prior on J also does the job of the old script's ridge penalty:
        # it keeps couplings finite when a pair is perfectly separated (a cell with zero counts).
        h = pm.Normal("h", mu=0.0, sigma=SIGMA_H, shape=k)
        J = pm.Normal("J", mu=0.0, sigma=SIGMA_J, shape=n_pairs)

        # ---- Energy of every possible state --------------------------------------------------
        # energy[s] = sum_d h_d * S[s,d]  +  sum_pairs J_p * SP[s,p]
        energy = pt.dot(S, h) + pt.dot(SP, J)          # shape (2^k,)

        # ---- log Z, computed stably ----------------------------------------------------------
        # Naively log(sum(exp(energy))) overflows. Subtract the max first, then add it back:
        #     log sum exp(e) = e_max + log sum exp(e - e_max)
        e_max = pt.max(energy)
        log_Z = e_max + pt.log(pt.sum(pt.exp(energy - e_max)))

        # ---- Log-likelihood via sufficient statistics ----------------------------------------
        # log L = N * ( h . m  +  J . C  -  log Z )
        # Every isolate is accounted for; none is looked at individually.
        log_lik = n_obs * (pt.dot(h, m) + pt.dot(J, C_vec) - log_Z)
        pm.Potential("ising_likelihood", log_lik)

        # Store logZ so we can reconstruct state probabilities later without recomputing.
        pm.Deterministic("log_Z", log_Z)

    return model


# =============================================================================================
# 5. POSTERIOR QUANTITIES
# =============================================================================================

def state_probabilities(h_draw, J_draw, S, SP):
    """Turn one posterior draw of (h, J) into a probability for every one of the 2^k states."""
    energy = S @ h_draw + SP @ J_draw
    energy -= energy.max()                 # same stability trick, now in numpy
    w = np.exp(energy)
    return w / w.sum()


def posterior_state_probs(idata, S, SP, n_sub=200, seed=SEED):
    """
    Compute state probabilities for a random subset of posterior draws.
    Returns (n_sub, 2^k). Subsampling keeps this cheap; 200 draws is plenty for the checks below.
    """
    h_all = idata.posterior["h"].values.reshape(-1, S.shape[1])
    J_all = idata.posterior["J"].values.reshape(-1, SP.shape[1])

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(h_all), size=min(n_sub, len(h_all)), replace=False)

    return np.array([state_probabilities(h_all[i], J_all[i], S, SP) for i in idx])


# =============================================================================================
# 6-8. THE CHECKS
# =============================================================================================

def check_moment_matching(W, S, SP, m, C_vec):
    """
    CHECK 1 

    A correctly fitted maxent model must reproduce the statistics it was built to match.
    If it does not, the fit has not converged, and any "leftover epistasis" you then measure is
    an artefact of the bad fit rather than a property of the biology. So this is a gate: look at
    it BEFORE looking at the triples.

    Because SP's columns are exactly the products x_i*x_j, "w @ SP" gives the model's pair rates.
    """
    model_m = W @ S            # (n_draws, k)
    model_C = W @ SP           # (n_draws, n_pairs)

    err_singles = np.abs(model_m.mean(axis=0) - m).max()
    err_pairs = np.abs(model_C.mean(axis=0) - C_vec).max()
    worst = max(err_singles, err_pairs)

    print("\n--- CHECK 1: does the model reproduce the singles and pairs it was fitted to? ---")
    print(f"  worst single-drug error : {err_singles:.5f}")
    print(f"  worst pair error        : {err_pairs:.5f}")
    print(f"  VERDICT: {'OK — fit is sound' if worst < 0.01 else 'FAILED — do not interpret triples'}")
    return worst


def check_all_orders(W, X, m, S, drug_names,
                     max_order=MAX_ORDER, max_tuples_per_order=MAX_TUPLES_PER_ORDER,
                     chunk=TUPLE_CHUNK, seed=SEED):
    """
    CHECK 2 — the pairwise-sufficiency test at EVERY interaction order.

    For each order o in 3, 4, ..., max_order (default up to k), it asks the same question check_triples
    asks for o=3: can a model that knows only singles and PAIRS predict the frequency with which all o
    drugs of a tuple are simultaneously resistant? If the pairwise model tracks the observed o-way
    frequencies at every order, there is no specific epistasis at ANY order — pairs are fully sufficient.

    HOW IT STAYS TRACTABLE
      - The number of o-tuples is C(k, o), which peaks near o = k/2; testing all orders touches ~2^k
        tuples in total. To bound cost, if C(k, o) > max_tuples_per_order we test a random (seeded)
        sample of that many tuples for that order. Set max_tuples_per_order high to be exhaustive.
      - Tuples are processed in chunks so we never hold a full (2^k x C(k,o)) matrix in memory: each
        chunk builds only a (2^k x chunk) product matrix, in float32.

      For a tuple t = (d1, ..., do):
        observed(t)   = mean over data of  x_d1 * ... * x_do           (empirical o-way co-resistance)
        pairwise(t)   = sum_states P_model(state) * [state resistant to all of t]   (per posterior draw)
        independent(t)= same under the product-of-marginals model                  (the null benchmark)

    RETURNS
      order_summary : DataFrame, one row per order, with the pairwise vs independent fit quality.
      tuple_details : DataFrame, one row per tested tuple (order, drugs, observed, pairwise +/- CrI,
                      independent, residual) — for drilling into which specific combinations, if any,
                      the pairwise model misses.

    READING THE OUTPUT
      pairwise RMSE ~ 0 and pairwise r ~ 1 at every order  => full pairwise sufficiency, no epistasis.
      A specific order where pairwise degrades (while independence is already poor) would be the
      signature of genuine higher-order epistasis at that order.
      CAVEAT: at high order, o-way co-resistance is rare, so observed frequencies are tiny and their
      variance is small; r and "variance explained" get unstable there. The robust quantity at high
      order is the MAX ABSOLUTE deviation |observed - pairwise| (reported), which stays interpretable.
    """
    Xv = X.values
    k = Xv.shape[1]
    from math import comb

    max_order = k if max_order is None else min(max_order, k)
    rng = np.random.default_rng(seed)

    # Independence-model weight of every state: P_ind(state) = prod_d m_d^x_d (1-m_d)^(1-x_d).
    p_ind = np.prod(S * m + (1 - S) * (1 - m), axis=1)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan

    order_rows = []
    tuple_rows = []

    print("\n--- CHECK 2 (all orders): does a pairs-only model predict co-resistance at every order? ---")
    header = f"  {'order':>5s} {'#tuples':>8s} {'ind_RMSE':>9s} {'pw_RMSE':>9s} {'pw_r':>7s} {'%var_expl':>9s} {'max|dev|':>9s} {'%in_CrI':>8s}"
    print(header)

    for o in range(3, max_order + 1):
        total = comb(k, o)
        all_tuples = itertools.combinations(range(k), o)

        # Sample tuples if this order has too many.
        if total > max_tuples_per_order:
            # reservoir-free sampling: materialise indices only (cheap ints), then pick.
            tuples = list(all_tuples)
            sel = rng.choice(len(tuples), size=max_tuples_per_order, replace=False)
            tuples = [tuples[i] for i in sel]
            sampled = True
        else:
            tuples = list(all_tuples)
            sampled = False

        # Accumulate predictions across chunks without holding everything at once.
        obs_all, pw_all, lo_all, hi_all, ind_all = [], [], [], [], []
        for start in range(0, len(tuples), chunk):
            batch = tuples[start:start + chunk]

            # observed o-way co-resistance for the batch
            obs = np.empty(len(batch))
            # state indicator: resistant to ALL drugs in the tuple, for every state
            ST = np.empty((S.shape[0], len(batch)), dtype=np.float32)
            for c, tup in enumerate(batch):
                prod_data = np.ones(Xv.shape[0])
                prod_state = np.ones(S.shape[0], dtype=np.float32)
                for d in tup:
                    prod_data = prod_data * Xv[:, d]
                    prod_state = prod_state * S[:, d].astype(np.float32)
                obs[c] = prod_data.mean()
                ST[:, c] = prod_state

            pred_draws = W @ ST                       # (n_draws, batch)
            pw = pred_draws.mean(axis=0)
            lo, hi = np.percentile(pred_draws, [3, 97], axis=0)
            ind = p_ind @ ST

            obs_all.append(obs); pw_all.append(pw); lo_all.append(lo); hi_all.append(hi); ind_all.append(ind)
            for c, tup in enumerate(batch):
                tuple_rows.append(dict(
                    order=o,
                    drugs=" & ".join(drug_names[d] for d in tup),
                    observed=obs[c], pairwise_pred=pw[c],
                    pairwise_lo=lo[c], pairwise_hi=hi[c],
                    independent_pred=ind[c], residual=obs[c] - pw[c],
                ))

        obs = np.concatenate(obs_all); pw = np.concatenate(pw_all)
        lo = np.concatenate(lo_all); hi = np.concatenate(hi_all); ind = np.concatenate(ind_all)

        resid = obs - pw
        var_expl = 1 - np.var(resid) / np.var(obs) if np.var(obs) > 0 else np.nan
        in_ci = np.mean((obs >= lo) & (obs <= hi))

        order_rows.append(dict(
            order=o, n_tuples=len(tuples), sampled=sampled,
            independent_rmse=rmse(obs, ind), independent_r=corr(obs, ind),
            pairwise_rmse=rmse(obs, pw), pairwise_r=corr(obs, pw),
            variance_explained=float(var_expl), max_abs_dev=float(np.abs(resid).max()),
            frac_inside_ci=float(in_ci),
        ))
        star = "*" if sampled else " "
        print(f"  {o:5d}{star}{len(tuples):8d} {rmse(obs, ind):9.4f} {rmse(obs, pw):9.4f} "
              f"{corr(obs, pw):7.3f} {100 * var_expl:8.1f}% {np.abs(resid).max():9.4f} {100 * in_ci:7.0f}%")

    print("  (* = order was subsampled to MAX_TUPLES_PER_ORDER)")
    print("  VERDICT: pairwise RMSE ~ 0 and max|dev| small at every order  =>  no epistasis at any order.")
    return pd.DataFrame(order_rows), pd.DataFrame(tuple_rows)


def report_couplings(idata, pairs, drug_names, top_n=15):
    """Print the strongest couplings — the biologically interpretable output."""
    J = idata.posterior["J"].values.reshape(-1, len(pairs))
    summary = pd.DataFrame({
        "pair": [f"{drug_names[i]} ~ {drug_names[j]}" for i, j in pairs],
        "J_mean": J.mean(axis=0),
        "J_lo": np.percentile(J, 3, axis=0),
        "J_hi": np.percentile(J, 97, axis=0),
    })
    summary["excludes_zero"] = (summary.J_lo > 0) | (summary.J_hi < 0)
    summary = summary.reindex(summary.J_mean.abs().sort_values(ascending=False).index)

    print(f"\n--- Strongest couplings (J > 0 co-resistance, J < 0 mutual exclusion) ---")
    print(summary.head(top_n).to_string(index=False,
                                        float_format=lambda v: f"{v:+.2f}"))
    return summary


# =============================================================================================
# 9. MAIN
# =============================================================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ---- Data -------------------------------------------------------------------------------
    X = load_binary_matrix(CSV_PATH)
    X = select_drugs(X, N_DRUGS)
    drug_names = list(X.columns)
    k = len(drug_names)

    m, C_vec, pairs = sufficient_statistics(X)
    S, SP = enumerate_states(k, pairs)

    print("\nSingle-drug non-susceptibility rates:")
    for name, rate in zip(drug_names, m):
        print(f"   {name:28s} {rate:.3f}")
        
    # idata=az.from_netcdf(OUT_DIR / "posterior.nc") 

    # ---- Fit --------------------------------------------------------------------------------
    model = build_model(m, C_vec, S, SP, n_obs=len(X))

    with model:
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            progressbar=True,
        )

    # ---- Sampler diagnostics ----------------------------------------------------------------
    # R-hat compares within- to between-chain variance; >1.01 means the chains disagree.
    # ESS is the effective number of independent draws; want a few hundred at least.
    summary = az.summary(idata, var_names=["h", "J"])
    print("\n--- Sampler diagnostics ---")
    print(f"  max R-hat : {summary['r_hat'].max():.4f}   (want < 1.01)")
    print(f"  min ESS   : {summary['ess_bulk'].min():.0f}   (want > 400)")
    n_div = int(idata.sample_stats["diverging"].values.sum())
    print(f"  divergences: {n_div}   (want 0; if many, raise TARGET_ACCEPT to 0.95)")

    # ---- The two checks ---------------------------------------------------------------------
    W = posterior_state_probs(idata, S, SP)

    worst = check_moment_matching(W, S, SP, m, C_vec)
    if worst >= 0.01:
        print("\nSTOPPING: the fit did not reproduce its own constraints, so the triple test")
        print("would be meaningless. Increase tuning, or reduce N_DRUGS, and refit.")
        return

    # Check tuples at every interaction order 3..k
    order_summary_df, tuple_details_df = check_all_orders(W, X, m, S, drug_names)

    coupling_df = report_couplings(idata, pairs, drug_names)

    # ---- Save -------------------------------------------------------------------------------
    idata.to_netcdf(OUT_DIR / "posterior.nc")           # reload with az.from_netcdf(...)
    order_summary_df.to_csv(OUT_DIR / "epistasis_by_order.csv", index=False)
    tuple_details_df.to_csv(OUT_DIR / "tuple_details.csv", index=False)
    coupling_df.to_csv(OUT_DIR / "couplings.csv", index=False)
    summary.to_csv(OUT_DIR / "parameter_summary.csv")
    print(f"\nWrote results to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
