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
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import proportional_hazard_test  # noqa: E402
from lifelines.utils import restricted_mean_survival_time  # noqa: E402
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
    # Unobserved individual differences in the propensity to leave (lognormal frailty). They
    # make any comparison that conditions on staying to month 12 non-randomised.
    frailty_sd: float = 0.5
    icc: float = 0.02
    icc_sensitivity: tuple[float, ...] = (0.005, 0.01, 0.02)
    bonus_dkk: int = 40_000
    followup_months: int = 24
    # Side effect built into the simulation: after the bonus is paid at month 12, treated
    # staff who postponed leaving do so in months 13 to 15.
    post_payment_hazard_mult: float = 1.8
    post_payment_months: tuple[int, int] = (13, 15)
    # Pre-specified decision threshold: at most this share of the 12 month gain may be lost
    # again in months 13 to 15.
    max_giveback_share: float = 0.5
    sample_sizes: tuple[int, ...] = field(default=(600, 1000, 1200, 1600, 2000, 2400, 3000))


A = Assumptions()


# ---------------------------------------------------------------- power and MDE


def _n_total(effect: float, p0: float, alpha: float, power: float, dof: int | None = None):
    """Total sample size (both arms, 1:1) for a two-sided test of p0 against p0 + effect,
    normal approximation with pooled variance under the null. With `dof`, t quantiles
    replace normal ones (used for cluster designs with few clusters)."""
    p1 = p0 + effect
    q = stats.norm if dof is None else stats.t(dof)
    z_a, z_b = q.ppf(1 - alpha / 2), q.ppf(power)
    pbar = (p0 + p1) / 2
    num = z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return 2 * (num / effect) ** 2


def n_required(effect: float, p0: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Total sample size, rounded up to whole people in each arm."""
    return int(2 * np.ceil(_n_total(effect, p0, alpha, power) / 2))


def _invert(n_eff: float, p0: float, alpha: float, power: float, dof: int | None) -> float:
    upper = (1 - p0) - 1e-9
    f = lambda d: _n_total(d, p0, alpha, power, dof) - n_eff  # noqa: E731
    if f(upper) > 0:
        return float("nan")  # not detectable even at 100% retention
    return float(optimize.brentq(f, 1e-4, upper))


def mde_individual(n_total: int, p0: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable increase in retention for a given total n (inverse of n_required).
    Stratification is ignored, which is slightly conservative."""
    return _invert(n_total, p0, alpha, power, None)


def mde_cluster(
    n_total: int, n_clusters: int, p0: float, icc: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """MDE when whole units are randomised. Same formula as mde_individual, applied to the
    effective sample size n / design effect, with t quantiles on (clusters - 2) degrees of
    freedom. Assumes equal cluster sizes; unequal units make the design effect larger."""
    deff = 1 + (n_total / n_clusters - 1) * icc
    return _invert(n_total / deff, p0, alpha, power, n_clusters - 2)


def power_table(a: Assumptions = A) -> pd.DataFrame:
    rows = []
    for n in a.sample_sizes:
        r = {"n_total": n, "mde_individual_pp": 100 * mde_individual(n, a.baseline_retention_12m)}
        for icc in (0.0, *a.icc_sensitivity):
            r[f"mde_cluster_icc_{icc}_pp"] = 100 * mde_cluster(
                n, a.n_units, a.baseline_retention_12m, icc
            )
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- simulation


def _retention(h: np.ndarray, months: int) -> np.ndarray:
    return (1 - np.clip(h, 0, 1)) ** months


def _population(n: int, a: Assumptions, rng: np.random.Generator) -> pd.DataFrame:
    unit = rng.integers(0, a.n_units, n)
    band = rng.choice(len(a.tenure_bands), n, p=a.tenure_shares)
    unit_mult = np.exp(rng.normal(0, a.unit_hazard_sd, a.n_units))
    frailty = np.exp(rng.normal(-(a.frailty_sd**2) / 2, a.frailty_sd, n))
    rel = np.asarray(a.tenure_hazard_mult)[band] * unit_mult[unit] * frailty
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
    df["retained_15m"] = ((df["event"] == 0) | (df["months"] > hi)).astype(int)
    df["retained_24m"] = (df["event"] == 0).astype(int)
    df["simulated"] = 1
    truth = {
        "baseline_monthly_hazard": h0,
        "hazard_ratio_months_1_12": hr,
        "hazard_ratio_months_13_15": a.post_payment_hazard_mult,
        "true_effect_retention_12m_pp": 100 * a.target_effect_pp,
    }
    return df.drop(columns="rel_hazard"), truth


# ---------------------------------------------------------------- planned analysis


def _lpm(df: pd.DataFrame, outcome: str) -> dict:
    """Linear probability model of `outcome` on treatment with stratum fixed effects, HC2."""
    m = smf.ols(f"{outcome} ~ treat + C(stratum)", data=df).fit(cov_type="HC2")
    ci = m.conf_int().loc["treat"]
    raw = df.groupby("treat")[outcome].mean()
    return {
        "effect_pp": 100 * m.params["treat"],
        "ci_low_pp": 100 * ci[0],
        "ci_high_pp": 100 * ci[1],
        "p_value": float(m.pvalues["treat"]),
        "mean_control": float(raw[0]),
        "mean_treated": float(raw[1]),
    }


def primary_analysis(df: pd.DataFrame) -> dict:
    """Pre-specified primary analysis: 12 month retention, all randomised staff."""
    return _lpm(df, "retained_12m")


def harm_analysis(df: pd.DataFrame) -> dict:
    """Pre-specified harm check on everyone randomised (no conditioning on later events).

    `left_m13_15` over all randomised staff equals the drop in retention from month 12 to
    month 15, so its treatment effect is exactly how much of the 12 month gain is lost again
    right after payment. Part of any such loss is mechanical: the bonus keeps people to month
    12 who are more likely to leave than average.
    """
    return _lpm(df, "left_m13_15")


def conditional_exit_rate(df: pd.DataFrame) -> dict:
    """Descriptive only: exit rate in months 13 to 15 among staff still employed at month 12.
    Not a randomised comparison, because the bonus changes who is still employed at 12."""
    at_risk = df[df["retained_12m"] == 1]
    rate = at_risk.groupby("treat")["left_m13_15"].mean()
    return {"rate_control": float(rate[0]), "rate_treated": float(rate[1])}


def retention_at(df: pd.DataFrame) -> dict:
    return {m: _lpm(df, f"retained_{m}") for m in ("15m", "24m")}


def _split_at_12(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["months", "event", "treat", "stratum"]].reset_index(names="id")
    first = d.assign(
        start=0,
        stop=np.minimum(d["months"], 12),
        event=np.where(d["months"] <= 12, d["event"], 0),
        post12=0,
    )
    second = d[d["months"] > 12].assign(start=12, stop=lambda x: x["months"], post12=1)
    ep = pd.concat([first, second], ignore_index=True)
    ep["treat_post12"] = ep["treat"] * ep["post12"]
    return ep[["id", "start", "stop", "event", "treat", "treat_post12", "stratum"]]


def cox_models(df: pd.DataFrame) -> dict:
    """Cox models stratified by randomisation stratum.

    * months_0_12: follow-up censored at month 12. A randomised comparison.
    * change_at_12: pre-specified test of a change in the hazard ratio at the payment date,
      using records split at month 12 with treat and treat x after-month-12. The hazard
      ratio after 12 is exp(treat + interaction). Hazard ratios after a point in time compare
      survivors, who are no longer randomised groups (Hernan 2010), so this is supporting
      evidence; the harm check above is the pre-specified test.
    * months_0_24: one pooled hazard ratio, with a proportional hazards test.
    """
    out = {}
    d = df[["months", "event", "treat", "stratum"]]
    first = d.assign(
        event=np.where(d["months"] <= 12, d["event"], 0), months=np.minimum(d["months"], 12)
    )
    for name, sub in (("months_0_12", first), ("months_0_24", d)):
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
    ep = _split_at_12(df)
    ctv = CoxTimeVaryingFitter().fit(
        ep, id_col="id", event_col="event", start_col="start", stop_col="stop", strata=["stratum"]
    )
    s = ctv.summary
    beta = ctv.params_
    cov = pd.DataFrame(np.asarray(ctv.variance_matrix_), index=beta.index, columns=beta.index)
    after = beta["treat"] + beta["treat_post12"]
    se_after = np.sqrt(
        cov.loc["treat", "treat"]
        + cov.loc["treat_post12", "treat_post12"]
        + 2 * cov.loc["treat", "treat_post12"]
    )
    out["change_at_12"] = {
        "hr_before_12": float(np.exp(beta["treat"])),
        "hr_after_12": float(np.exp(after)),
        "hr_after_12_ci": [
            float(np.exp(after - 1.96 * se_after)),
            float(np.exp(after + 1.96 * se_after)),
        ],
        "ratio_after_vs_before": float(s.loc["treat_post12", "exp(coef)"]),
        "ratio_ci_low": float(s.loc["treat_post12", "exp(coef) lower 95%"]),
        "ratio_ci_high": float(s.loc["treat_post12", "exp(coef) upper 95%"]),
        "ratio_p_value": float(s.loc["treat_post12", "p"]),
    }
    return out


def km_plot(df: pd.DataFrame, path: Path, a: Assumptions = A) -> dict:
    fig, ax = plt.subplots(figsize=(6.8, 4))
    km_at, rmst = {}, {}
    for arm, lab in ((0, "control"), (1, "bonus")):
        sub = df[df["treat"] == arm]
        km = KaplanMeierFitter().fit(sub["months"], sub["event"], label=lab)
        km.plot_survival_function(ax=ax, ci_show=True)
        km_at[lab] = {str(m): float(km.survival_function_at_times(m).iloc[0]) for m in (12, 24)}
        rmst[lab] = float(restricted_mean_survival_time(km, t=a.followup_months))
    ax.axvline(12, color="grey", ls=":", lw=1)
    ax.text(12.2, ax.get_ylim()[0] + 0.01, "bonus paid", fontsize=8, color="grey")
    ax.set_xlabel("months since randomisation")
    ax.set_ylabel("share still employed")
    ax.set_title(f"Kaplan-Meier, {LABEL}", fontsize=8)
    ax.set_xlim(0, a.followup_months)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {
        "km": km_at,
        "rmst_24m_months": rmst,
        "rmst_diff_months": rmst["bonus"] - rmst["control"],
    }


def simulated_rejection_rates(n: int, reps: int, seed: int, a: Assumptions = A) -> dict:
    """Monte Carlo operating characteristics of the planned tests (LPMs only, fast):
    power of the primary test under the assumed effect, power of the harm check with the
    built-in spike, and its rejection rate when there is no spike."""
    no_spike = Assumptions(**{**asdict(a), "post_payment_hazard_mult": 1.0})
    hits = {"primary_power": 0, "harm_power": 0, "harm_rejections_without_spike": 0}
    for r in range(reps):
        df, _ = simulate(n, seed + r, a)
        hits["primary_power"] += primary_analysis(df)["p_value"] < a.alpha
        hits["harm_power"] += harm_analysis(df)["p_value"] < a.alpha
        df0, _ = simulate(n, seed + r, no_spike)
        hits["harm_rejections_without_spike"] += harm_analysis(df0)["p_value"] < a.alpha
    out = {k: v / reps for k, v in hits.items()}
    out.update({f"{k}_mc_se": float(np.sqrt(v * (1 - v) / reps)) for k, v in list(out.items())})
    out["reps"] = reps
    return out


def implied_icc(a: Assumptions = A, seeds: int = 30, n: int = 6000) -> float:
    """ICC of 12 month retention between units implied by the simulation (one way ANOVA
    estimator on control staff, averaged over independent draws of the unit effects)."""
    vals = []
    for s in range(seeds):
        df, _ = simulate(n, 90_000 + s, a)
        c = df[df["treat"] == 0]
        g = c.groupby("unit")["retained_12m"]
        k, m = g.ngroups, g.size().mean()
        msb = (g.size() * (g.mean() - c["retained_12m"].mean()) ** 2).sum() / (k - 1)
        msw = (g.var(ddof=1) * (g.size() - 1)).sum() / (len(c) - k)
        vals.append((msb - msw) / (msb + (m - 1) * msw))
    return float(np.mean(vals))


def run(seed: int = 20261201, mc_reps: int = 400, out: Path = OUT, a: Assumptions = A):
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
    n_t = int((df.treat == 1).sum())
    paid = int(df.loc[df.treat == 1, "retained_12m"].sum())
    primary, harm = primary_analysis(df), harm_analysis(df)
    surv = km_plot(df, out / "km_simulated.png", a)
    giveback = harm["effect_pp"] / primary["effect_pp"] if primary["effect_pp"] > 0 else None
    p0, p1 = a.baseline_retention_12m, a.baseline_retention_12m + a.target_effect_pp
    res = {
        "label": LABEL,
        "assumptions": asdict(a),
        "n_required_for_target": n_req,
        "n_pilot": n_pilot,
        "n_treated": n_t,
        "n_strata": int(df["stratum"].nunique()),
        "mde_at_pilot_n_pp": 100 * mde_individual(n_pilot, p0),
        "mde_cluster_at_pilot_n_pp": {
            str(icc): 100 * mde_cluster(n_pilot, a.n_units, p0, icc)
            for icc in (0.0, *a.icc_sensitivity)
        },
        "implied_icc_in_simulation": implied_icc(a),
        "operating_characteristics": simulated_rejection_rates(n_pilot, mc_reps, seed + 1000, a),
        "truth": truth,
        "primary": primary,
        "harm_m13_15": harm,
        "giveback_share": giveback,
        "conditional_exit_rate_m13_15": conditional_exit_rate(df),
        "retention": retention_at(df),
        "cox": cox_models(df),
        "survival": surv,
        "decision_rule": {
            "primary_lower_ci_above_zero": bool(primary["ci_low_pp"] > 0),
            "giveback_share_below_max": bool(
                giveback is not None and giveback <= a.max_giveback_share
            ),
            "cost_test": "needs a replacement cost estimate, not available",
        },
        "cost": {
            "bonuses_paid": paid,
            "bonus_cost_dkk": paid * a.bonus_dkk,
            "planned_share_paid": p1,
            "planned_n_paid": p1 * (n_pilot // 2),
            "planned_budget_dkk": a.bonus_dkk * p1 * (n_pilot // 2),
            "planned_cost_per_extra_retained_12m_dkk": a.bonus_dkk * p1 / a.target_effect_pp,
            "simulated_cost_per_extra_person_month_24m_dkk": (
                paid * a.bonus_dkk / (n_t * surv["rmst_diff_months"])
                if surv["rmst_diff_months"] > 0
                else None
            ),
        },
        "seed": seed,
    }
    (out / "pilot_results.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("n_pilot", "primary", "harm_m13_15")}, indent=1))
