"""Estimators of the average effect of training on the treated (ATT) on 1978 earnings.

All observational estimators below target the ATT, the same quantity the NSW experiment
identifies for its participants, so each can be compared directly with the experimental
benchmark.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from .data import COVARIATES, OUTCOME

METHODS = {
    "diff_means": "Difference in means",
    "ols": "OLS with covariates",
    "psm": "Propensity score matching (1 NN)",
    "ipw": "Inverse probability weighting",
    "aipw": "Doubly robust (AIPW)",
}

FAMILY = sm.families.Binomial()
OLS_COLS = COVARIATES + ["u74", "u75"]


def ps_design(df: pd.DataFrame) -> np.ndarray:
    """Propensity score design in the spirit of Dehejia and Wahba (1999).

    One fixed specification is used for both observational samples so the comparison is
    not tuned per sample: linear terms, squares of age, education and prior earnings,
    unemployment indicators and an education by 1974 earnings interaction.
    """
    a, e = df["age"], df["education"]
    r4, r5 = df["re74"] / 1000, df["re75"] / 1000
    cols = [
        a,
        a**2,
        e,
        e**2,
        df["black"],
        df["hispanic"],
        df["married"],
        df["nodegree"],
        r4,
        r5,
        r4**2,
        r5**2,
        df["u74"],
        df["u75"],
        e * r4,
    ]
    return np.column_stack([np.asarray(c, dtype=float) for c in cols])


def ps_design_simple(df: pd.DataFrame) -> np.ndarray:
    """Linear terms only. Used as a sensitivity check on the propensity specification."""
    return df[COVARIATES].to_numpy(dtype=float)


def propensity(df: pd.DataFrame, spec: str = "dw") -> np.ndarray:
    """Unpenalised logistic regression (maximum likelihood by IRLS) on standardised features.

    Standardising only rescales the coefficients; the fitted probabilities are the same as
    R's glm(binomial) on the raw design, which r/crosscheck.R uses to check this.
    """
    x = ps_design(df) if spec == "dw" else ps_design_simple(df)
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    with warnings.catch_warnings():
        # Bootstrap resamples of the PSID sample can be close to separated; the fit still
        # returns finite probabilities, which are clipped below.
        warnings.simplefilter("ignore")
        res = sm.GLM(
            df["treat"].to_numpy(), sm.add_constant(z, has_constant="add"), family=FAMILY
        ).fit(tol=1e-10, maxiter=200)
        p = np.asarray(res.fittedvalues)
    return np.clip(p, 1e-6, 1 - 1e-6)


def _ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xd = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
    return beta


def att_weights(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ATT weights: 1 for treated, p / (1 - p) for controls (normalised later)."""
    return np.where(t == 1, 1.0, p / (1 - p))


@dataclass
class Fit:
    estimates: dict[str, float]
    ps: np.ndarray
    match_idx: np.ndarray  # index (into df) of the matched control for each treated unit
    weights: np.ndarray


def estimate_all(df: pd.DataFrame, spec: str = "dw") -> Fit:
    y = df[OUTCOME].to_numpy()
    t = df["treat"].to_numpy()
    tr, co = t == 1, t == 0
    est: dict[str, float] = {}

    est["diff_means"] = y[tr].mean() - y[co].mean()

    x = df[OLS_COLS].to_numpy()
    beta = _ols_fit(np.column_stack([t, x]), y)
    est["ols"] = beta[1]

    p = propensity(df, spec)
    lp = np.log(p / (1 - p))

    # 1 nearest neighbour matching on the logit propensity score, with replacement.
    co_idx = np.flatnonzero(co)
    nn = NearestNeighbors(n_neighbors=1).fit(lp[co_idx, None])
    _, j = nn.kneighbors(lp[tr, None])
    match_idx = co_idx[j[:, 0]]
    est["psm"] = (y[tr] - y[match_idx]).mean()

    w = att_weights(p, t)
    wc = w[co] / w[co].sum()
    est["ipw"] = y[tr].mean() - (wc * y[co]).sum()

    # AIPW for the ATT: outcome model for controls, residuals reweighted.
    b0 = _ols_fit(x[co], y[co])
    m0 = b0[0] + x @ b0[1:]
    res = y - m0
    est["aipw"] = res[tr].mean() - (wc * res[co]).sum()

    return Fit({k: float(v) for k, v in est.items()}, p, match_idx, w)


def bootstrap(df: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Nonparametric bootstrap, resampling treated and controls separately.

    Note: Abadie and Imbens (2008) show the naive bootstrap is not valid for nearest
    neighbour matching, so the matching interval is indicative only.
    """
    rng = np.random.default_rng(seed)
    tr_idx = np.flatnonzero(df["treat"].to_numpy() == 1)
    co_idx = np.flatnonzero(df["treat"].to_numpy() == 0)
    rows = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(tr_idx, tr_idx.size), rng.choice(co_idx, co_idx.size)])
        rows.append(estimate_all(df.iloc[idx].reset_index(drop=True)).estimates)
    return pd.DataFrame(rows)


def welch_ci(df: pd.DataFrame, level: float = 0.95) -> tuple[float, float]:
    y, t = df[OUTCOME].to_numpy(), df["treat"].to_numpy()
    a, b = y[t == 1], y[t == 0]
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    q = stats.t.ppf(0.5 + level / 2, dof)
    d = a.mean() - b.mean()
    return float(d - q * se), float(d + q * se)


def smd(df: pd.DataFrame, cols: list[str], weights: np.ndarray | None = None) -> pd.Series:
    """Standardised mean difference (treated minus controls) with a pooled unweighted SD.

    If weights are given, the control mean is weighted (treated weights are ignored).
    """
    t = df["treat"].to_numpy() == 1
    out = {}
    for c in cols:
        v = df[c].to_numpy()
        sd = np.sqrt((v[t].var(ddof=1) + v[~t].var(ddof=1)) / 2)
        mc = v[~t].mean() if weights is None else np.average(v[~t], weights=weights[~t])
        out[c] = (v[t].mean() - mc) / sd if sd > 0 else 0.0
    return pd.Series(out)


def effective_n(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())
