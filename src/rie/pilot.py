"""Design and planned analysis for a HYPOTHETICAL retention bonus pilot.

Everything in this module that looks like staff data is SIMULATED. No Forsvaret, Danish
register or any other real personnel data is used. The simulation exists only to show that
the planned analysis runs end to end and recovers effects that are known by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test, proportional_hazard_test  # noqa: E402
from scipy import optimize, stats  # noqa: E402

from .data import ROOT  # noqa: E402

OUT = ROOT / "design"
LABEL = "SIMULATED DATA (hypothetical pilot, not real personnel data)"


@dataclass(frozen=True)
class Assumptions:
    """Planning assumptions. All are hypothetical and chosen for illustration."""

    baseline_retention_12m: float = 0.80
    target_effect_pp: float = 0.05
    alpha: float = 0.05
    power: float = 0.80
    n_units: int = 12
    tenure_bands: tuple[str, ...] = ("0-3y", "3-8y", "8y+")
    tenure_hazard_mult: tuple[float, ...] = (1.6, 1.0, 0.6)
    tenure_shares: tuple[float, ...] = (0.40, 0.35, 0.25)
    unit_hazard_sd: float = 0.20
    icc: float = 0.02
    bonus_dkk: int = 40_000
    followup_months: int = 24
    # Side effect built into the simulation: after the bonus is paid at month 12, treated
    # staff who postponed leaving do so in months 13 to 15.
    post_payment_hazard_mult: float = 1.8
    post_payment_months: tuple[int, int] = (13, 15)
    sample_sizes: tuple[int, ...] = field(default=(600, 1000, 1200, 1600, 2000, 2400, 3000))


A = Assumptions()


# ---------------------------------------------------------------- power and MDE


def _n_total(effect: float, p0: float, alpha: float, power: float) -> float:
    """Total sample size (both arms, 1:1) for a two-sided test of p0 against p0 + effect,
    normal approximation with pooled variance under the null."""
    p1 = p0 + effect
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p0 + p1) / 2
    num = z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return 2 * (num / effect) ** 2


def n_required(effect: float, p0: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Total sample size, rounded up to whole people in each arm."""
    return int(2 * np.ceil(_n_total(effect, p0, alpha, power) / 2))


def mde_individual(n_total: int, p0: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable increase in retention for a given total n (inverse of n_required).
    Stratification is ignored, which is slightly conservative."""
    upper = (1 - p0) - 1e-9
    return float(optimize.brentq(lambda d: _n_total(d, p0, alpha, power) - n_total, 1e-4, upper))


def mde_cluster(n_total: int, n_clusters: int, p0: float, icc: float, **kw) -> float:
    """MDE when whole units are randomised: inflate the variance by the design effect and use
    a t quantile with (clusters - 2) degrees of freedom."""
    m = n_total / n_clusters
    deff = 1 + (m - 1) * icc
    alpha, power = kw.get("alpha", 0.05), kw.get("power", 0.80)
    dof = n_clusters - 2
    z = stats.t.ppf(1 - alpha / 2, dof) + stats.t.ppf(power, dof)
    return float(z * np.sqrt(deff * 2 * p0 * (1 - p0) / (n_total / 2)))


def power_table(a: Assumptions = A) -> pd.DataFrame:
    rows = []
    for n in a.sample_sizes:
        rows.append(
            {
                "n_total": n,
                "mde_individual_pp": 100 * mde_individual(n, a.baseline_retention_12m),
                "mde_cluster_pp": 100 * mde_cluster(n, a.n_units, a.baseline_retention_12m, a.icc),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- simulation


def _retention(h: np.ndarray, months: int) -> np.ndarray:
    return (1 - np.clip(h, 0, 1)) ** months


def _population(n: int, a: Assumptions, rng: np.random.Generator) -> pd.DataFrame:
    unit = rng.integers(0, a.n_units, n)
    band = rng.choice(len(a.tenure_bands), n, p=a.tenure_shares)
    unit_mult = np.exp(rng.normal(0, a.unit_hazard_sd, a.n_units))
    rel = np.asarray(a.tenure_hazard_mult)[band] * unit_mult[unit]
    age = np.round(22 + 6 * band + rng.gamma(2.0, 3.0, n)).astype(int)
    return pd.DataFrame(
        {
            "unit": unit + 1,
            "tenure_band": np.asarray(a.tenure_bands)[band],
            "age": age,
            "rel_hazard": rel,
        }
    )


def calibrate(rel: np.ndarray, a: Assumptions) -> tuple[float, float]:
    """Return (baseline monthly hazard, pre payment hazard ratio) so that expected 12 month
    retention is the baseline in control and baseline + target effect under treatment."""
    p0, p1 = a.baseline_retention_12m, a.baseline_retention_12m + a.target_effect_pp
    h0 = optimize.brentq(lambda h: _retention(h * rel, 12).mean() - p0, 1e-6, 0.5)
    hr = optimize.brentq(lambda r: _retention(h0 * r * rel, 12).mean() - p1, 1e-3, 1.0)
    return float(h0), float(hr)


def stratified_assignment(strata: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """1:1 allocation within each stratum (random permutation of a balanced vector)."""
    t = np.zeros(len(strata), dtype=int)
    for _, idx in strata.groupby(strata).groups.items():
        idx = np.asarray(list(idx))
        k = len(idx)
        arm = np.array([1] * (k // 2) + [0] * (k - k // 2))
        if k % 2 == 1 and rng.random() < 0.5:
            arm = 1 - arm
        t[idx] = rng.permutation(arm)
    return t


def simulate(n: int, seed: int, a: Assumptions = A) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    df = _population(n, a, rng)
    df["stratum"] = "u" + df["unit"].astype(str) + "_" + df["tenure_band"]
    df["treat"] = stratified_assignment(df["stratum"], rng)
    h0, hr = calibrate(df["rel_hazard"].to_numpy(), a)
    h = h0 * df["rel_hazard"].to_numpy()
    t = df["treat"].to_numpy()
    months = np.arange(1, a.followup_months + 1)
    lo, hi = a.post_payment_months
    mult = np.ones((n, months.size))
    mult[t == 1, :12] = hr
    mult[np.ix_(t == 1, (months >= lo) & (months <= hi))] = a.post_payment_hazard_mult
    haz = np.clip(h[:, None] * mult, 0, 1)
    exits = rng.random(haz.shape) < haz
    first = np.where(exits.any(1), exits.argmax(1) + 1, a.followup_months)
    df["event"] = exits.any(1).astype(int)
    df["months"] = first
    df["retained_12m"] = ((df["event"] == 0) | (df["months"] > 12)).astype(int)
    df["left_m13_15"] = ((df["event"] == 1) & df["months"].between(lo, hi)).astype(int)
    df["simulated"] = 1
    truth = {
        "baseline_monthly_hazard": h0,
        "hazard_ratio_months_1_12": hr,
        "hazard_ratio_months_13_15": a.post_payment_hazard_mult,
        "true_effect_retention_12m_pp": 100 * a.target_effect_pp,
    }
    return df.drop(columns="rel_hazard"), truth


# ---------------------------------------------------------------- planned analysis


def primary_analysis(df: pd.DataFrame) -> dict:
    """Pre-specified primary analysis: linear probability model of 12 month retention on
    treatment with stratum fixed effects and HC2 robust standard errors."""
    m = smf.ols("retained_12m ~ treat + C(stratum)", data=df).fit(cov_type="HC2")
    ci = m.conf_int().loc["treat"]
    raw = df.groupby("treat")["retained_12m"].mean()
    return {
        "effect_pp": 100 * m.params["treat"],
        "ci_low_pp": 100 * ci[0],
        "ci_high_pp": 100 * ci[1],
        "p_value": float(m.pvalues["treat"]),
        "retention_control": float(raw[0]),
        "retention_treated": float(raw[1]),
    }


def side_effect_analysis(df: pd.DataFrame) -> dict:
    """Pre-specified harm check: exits in months 13 to 15, among those still employed at 12."""
    at_risk = df[df["retained_12m"] == 1]
    m = smf.ols("left_m13_15 ~ treat + C(stratum)", data=at_risk).fit(cov_type="HC2")
    ci = m.conf_int().loc["treat"]
    rate = at_risk.groupby("treat")["left_m13_15"].mean()
    return {
        "effect_pp": 100 * m.params["treat"],
        "ci_low_pp": 100 * ci[0],
        "ci_high_pp": 100 * ci[1],
        "p_value": float(m.pvalues["treat"]),
        "rate_control": float(rate[0]),
        "rate_treated": float(rate[1]),
    }


def retention_24m(df: pd.DataFrame) -> dict:
    r = df.assign(ret=1 - df["event"]).groupby("treat")["ret"].mean()
    return {"control": float(r[0]), "treated": float(r[1]), "diff_pp": 100 * (r[1] - r[0])}


def cox_models(df: pd.DataFrame) -> dict:
    """Cox models stratified by randomisation stratum: months 0 to 12, months 12 to 24
    (conditional on staying to 12), and the full period with a proportional hazards test."""
    out = {}
    d = df[["months", "event", "treat", "stratum"]]
    first = d.assign(
        event=np.where(d["months"] <= 12, d["event"], 0), months=np.minimum(d["months"], 12)
    )
    second = d[d["months"] > 12].assign(months=lambda x: x["months"] - 12)
    for name, sub in (("months_0_12", first), ("months_13_24", second), ("months_0_24", d)):
        cph = CoxPHFitter().fit(sub, "months", "event", strata=["stratum"])
        s = cph.summary.loc["treat"]
        out[name] = {
            "hazard_ratio": float(s["exp(coef)"]),
            "ci_low": float(s["exp(coef) lower 95%"]),
            "ci_high": float(s["exp(coef) upper 95%"]),
            "p_value": float(s["p"]),
            "n": int(len(sub)),
            "events": int(sub["event"].sum()),
        }
        if name == "months_0_24":
            ph = proportional_hazard_test(cph, sub, time_transform="rank")
            out[name]["ph_test_p"] = float(ph.summary.loc["treat", "p"])
    return out


def km_plot(df: pd.DataFrame, path: Path, a: Assumptions = A) -> dict:
    fig, ax = plt.subplots(figsize=(6.8, 4))
    km_at = {}
    for arm, lab in ((0, "control"), (1, "bonus")):
        sub = df[df["treat"] == arm]
        km = KaplanMeierFitter().fit(sub["months"], sub["event"], label=lab)
        km.plot_survival_function(ax=ax, ci_show=True)
        km_at[lab] = {str(m): float(km.survival_function_at_times(m).iloc[0]) for m in (12, 24)}
    lr = logrank_test(
        df.loc[df.treat == 1, "months"],
        df.loc[df.treat == 0, "months"],
        df.loc[df.treat == 1, "event"],
        df.loc[df.treat == 0, "event"],
    )
    ax.axvline(12, color="grey", ls=":", lw=1)
    ax.text(12.2, ax.get_ylim()[0] + 0.01, "bonus paid", fontsize=8, color="grey")
    ax.set_xlabel("months since randomisation")
    ax.set_ylabel("share still employed")
    ax.set_title(f"Kaplan-Meier, {LABEL}", fontsize=8)
    ax.set_xlim(0, a.followup_months)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"km": km_at, "logrank_p_0_24": float(lr.p_value)}


def simulated_power(n: int, reps: int, seed: int, a: Assumptions = A) -> float:
    """Share of simulated pilots where the primary analysis rejects at alpha."""
    hits = 0
    for r in range(reps):
        df, _ = simulate(n, seed + r, a)
        hits += primary_analysis(df)["p_value"] < a.alpha
    return hits / reps


def run(seed: int = 20261201, power_reps: int = 200, out: Path = OUT, a: Assumptions = A):
    out.mkdir(parents=True, exist_ok=True)
    pt = power_table(a)
    pt.round(2).to_csv(out / "power_table.csv", index=False)
    n_req = n_required(a.target_effect_pp, a.baseline_retention_12m, a.alpha, a.power)
    n_pilot = int(np.ceil(n_req / 100) * 100)
    df, truth = simulate(n_pilot, seed, a)
    df.to_csv(out / "simulated_pilot_data.csv", index=False)
    balance = (
        df.groupby("treat")
        .agg(n=("treat", "size"), mean_age=("age", "mean"))
        .join(pd.crosstab(df["treat"], df["tenure_band"], normalize="index"))
    )
    balance.round(3).to_csv(out / "simulated_baseline_balance.csv")
    paid = int(df.loc[df.treat == 1, "retained_12m"].sum())
    n_t = int((df.treat == 1).sum())
    res = {
        "label": LABEL,
        "assumptions": asdict(a),
        "n_required_for_target": n_req,
        "n_pilot": n_pilot,
        "n_treated": n_t,
        "n_strata": int(df["stratum"].nunique()),
        "mde_at_pilot_n_pp": 100 * mde_individual(n_pilot, a.baseline_retention_12m),
        "mde_cluster_at_pilot_n_pp": 100
        * mde_cluster(n_pilot, a.n_units, a.baseline_retention_12m, a.icc),
        "simulated_power_at_pilot_n": simulated_power(n_pilot, power_reps, seed + 1000, a),
        "power_reps": power_reps,
        "truth": truth,
        "primary": primary_analysis(df),
        "side_effect_m13_15": side_effect_analysis(df),
        "retention_24m": retention_24m(df),
        "cox": cox_models(df),
        "survival": km_plot(df, out / "km_simulated.png", a),
        "cost": {
            "bonuses_paid": paid,
            "bonus_cost_dkk": paid * a.bonus_dkk,
            "planned_cost_per_extra_retained_dkk": a.bonus_dkk
            * (a.baseline_retention_12m + a.target_effect_pp)
            / a.target_effect_pp,
            "planned_budget_dkk": a.bonus_dkk
            * (a.baseline_retention_12m + a.target_effect_pp)
            * (n_pilot // 2),
        },
        "seed": seed,
    }
    (out / "pilot_results.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("n_pilot", "primary", "side_effect_m13_15")}, indent=1))
