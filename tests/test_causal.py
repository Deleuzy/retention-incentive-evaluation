"""Offline tests on the cached raw files in data/raw (no network)."""

import json

import numpy as np
import pandas as pd
import pytest

from rie import data
from rie import estimators as E


@pytest.fixture(scope="module")
def samples():
    paths = data.fetch(download=False)
    assert set(paths) == set(data.SOURCES)
    return {s: data.load(s) for s in data.SAMPLES}


def test_checksums_and_sizes(samples):
    exp = samples["experimental"]
    assert (exp.treat == 1).sum() == 185 and (exp.treat == 0).sum() == 260
    assert len(samples["psid"]) == 185 + 2490
    assert len(samples["cps"]) == 185 + 15992


def test_checksum_mismatch_is_rejected(tmp_path):
    for name in data.SOURCES:
        (tmp_path / name).write_bytes(b"not the real file")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.fetch(tmp_path, download=False)


def test_experimental_benchmark_matches_published_value(samples):
    # Dehejia and Wahba (1999) report 1,794 USD for their NSW subsample.
    est = E.estimate_all(samples["experimental"]).estimates
    assert est["diff_means"] == pytest.approx(1794.34, abs=0.01)
    lo, hi = E.welch_ci(samples["experimental"])
    assert lo < 1794.34 < hi


def test_all_estimators_agree_on_randomised_data(samples):
    est = E.estimate_all(samples["experimental"]).estimates
    for m in ("ols", "ipw", "aipw"):
        assert abs(est[m] - est["diff_means"]) < 250, m


def test_naive_observational_comparison_has_wrong_sign(samples):
    for s in ("psid", "cps"):
        assert E.estimate_all(samples[s]).estimates["diff_means"] < -5000


def test_ipw_improves_balance(samples):
    df = samples["cps"]
    fit = E.estimate_all(df)
    raw = E.smd(df, E.OLS_COLS).abs().max()
    weighted = E.smd(df, E.OLS_COLS, fit.weights).abs().max()
    assert weighted < raw / 5


def test_att_weights():
    p = np.array([0.5, 0.2, 0.8])
    t = np.array([1, 0, 0])
    assert np.allclose(E.att_weights(p, t), [1.0, 0.25, 4.0])


def _synthetic_covariates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "age": rng.normal(30, 8, n),
            "education": rng.integers(8, 16, n).astype(float),
            "black": rng.integers(0, 2, n).astype(float),
            "hispanic": np.zeros(n),
            "married": rng.integers(0, 2, n).astype(float),
            "nodegree": rng.integers(0, 2, n).astype(float),
            "re74": np.maximum(0, rng.normal(8000, 6000, n)),
        }
    )
    df["re75"] = np.maximum(0, df["re74"] + rng.normal(0, 2000, n))
    return data.add_features(df)


TRUE_ATT = 1500.0


def test_estimators_recover_known_att_when_both_models_are_right():
    rng = np.random.default_rng(0)
    df = _synthetic_covariates(4000, rng)
    logit = 1.5 - 0.0004 * df["re75"] + 0.5 * df["u75"]
    df["treat"] = (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)
    df["re78"] = 2000 + 0.8 * df["re75"] + TRUE_ATT * df["treat"] + rng.normal(0, 3000, len(df))
    est = E.estimate_all(df).estimates
    assert abs(est["diff_means"] - TRUE_ATT) > 1500
    for m in ("ols", "ipw", "aipw"):
        assert est[m] == pytest.approx(TRUE_ATT, abs=400), m
    assert est["psm"] == pytest.approx(TRUE_ATT, abs=700)


def test_aipw_is_doubly_robust_wrong_propensity_model():
    """Selection through an interaction the propensity design lacks; outcome model right."""
    rng = np.random.default_rng(1)
    df = _synthetic_covariates(6000, rng)
    low_single = ((df["re75"] < 6000) & (df["married"] == 0)).astype(float)
    older_black = ((df["age"] > 35) & (df["black"] == 1)).astype(float)
    logit = -0.5 + 2.0 * low_single - 1.5 * older_black
    df["treat"] = (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)
    noise = rng.normal(0, 3000, len(df))
    df["re78"] = 2000 + 0.8 * df["re75"] + 300 * df["married"] + TRUE_ATT * df["treat"] + noise
    est = E.estimate_all(df).estimates
    assert est["aipw"] == pytest.approx(TRUE_ATT, abs=250)
    assert abs(est["ipw"] - TRUE_ATT) > abs(est["aipw"] - TRUE_ATT)


def test_aipw_is_doubly_robust_wrong_outcome_model():
    """Outcome quadratic in 1975 earnings (OLS is linear); propensity model right."""
    rng = np.random.default_rng(2)
    df = _synthetic_covariates(6000, rng)
    logit = 1.5 - 0.0004 * df["re75"] + 0.5 * df["u75"]
    df["treat"] = (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)
    noise = rng.normal(0, 3000, len(df))
    df["re78"] = 2000 + 0.3 * df["re75"] + 6e-5 * df["re75"] ** 2 + TRUE_ATT * df["treat"] + noise
    est = E.estimate_all(df).estimates
    assert abs(est["ols"] - TRUE_ATT) > 1000
    assert est["aipw"] == pytest.approx(TRUE_ATT, abs=400)


def test_joint_bootstrap_shares_treated_and_keeps_group_sizes(samples):
    sub = {k: samples[k] for k in ("experimental", "psid")}
    boot = E.bootstrap_joint(sub, reps=3, seed=4)
    assert set(boot["sample"]) == set(sub) and boot["rep"].nunique() == 3
    assert len(boot) == 3 * 2 * len(E.METHODS)
    bad = {"experimental": samples["experimental"], "cps": samples["cps"].iloc[1:]}
    with pytest.raises(ValueError, match="treated units"):
        E.bootstrap_joint(bad, reps=1, seed=0)


def test_psid_weighting_balance_failure_is_reported(samples):
    """Weighting on PSID does not balance the covariates; the results must say so."""
    df = samples["psid"]
    fit = E.estimate_all(df)
    sd = E.pooled_sd(df, E.OLS_COLS)
    assert E.smd(df, E.OLS_COLS, fit.weights, sd_ref=sd).abs().max() > 0.1
    assert "fails its own diagnostics" in (data.ROOT / "causal" / "RESULTS.md").read_text()


def test_bootstrap_shape_and_reproducibility(samples):
    a = E.bootstrap(samples["experimental"], reps=5, seed=1)
    b = E.bootstrap(samples["experimental"], reps=5, seed=1)
    assert list(a.columns) == list(E.METHODS)
    pd.testing.assert_frame_equal(a, b)


def test_smd_fixed_denominator():
    df = pd.DataFrame({"treat": [1, 1, 0, 0], "x": [1.0, 3.0, 0.0, 2.0]})
    ref = pd.Series({"x": 2.0})
    assert E.smd(df, ["x"], sd_ref=ref)["x"] == pytest.approx(0.5)


def test_committed_estimates_reproduce_point_estimates(samples):
    committed = pd.read_csv(data.ROOT / "causal" / "estimates.csv")
    for s, df in samples.items():
        est = E.estimate_all(df).estimates
        for m, v in est.items():
            c = committed[(committed["sample"] == s) & (committed["method"] == m)]
            assert c["estimate"].iloc[0] == pytest.approx(v, abs=0.01), (s, m)


def test_r_crosscheck_agrees_with_python():
    r = json.loads((data.ROOT / "r" / "r_crosscheck.json").read_text())
    summary = json.loads((data.ROOT / "causal" / "summary.json").read_text())
    est = pd.read_csv(data.ROOT / "causal" / "estimates.csv").set_index(["sample", "method"])
    pilot = json.loads((data.ROOT / "design" / "pilot_results.json").read_text())
    assert r["benchmark"] == pytest.approx(summary["benchmark"], abs=1e-3)
    assert r["benchmark_welch_ci"] == pytest.approx(summary["benchmark_welch_ci"], abs=1e-3)
    assert r["ipw_psid"] == pytest.approx(est.loc[("psid", "ipw"), "estimate"], abs=1.0)
    assert r["ipw_cps"] == pytest.approx(est.loc[("cps", "ipw"), "estimate"], abs=1.0)
    hr = pilot["cox"]["months_0_12"]["hazard_ratio"]
    assert r["cox_hr_months_0_12_simulated"] == pytest.approx(hr, rel=1e-3)
    ratio = pilot["cox"]["change_at_12"]["ratio_after_vs_before"]
    assert r["cox_change_at_12_ratio_simulated"] == pytest.approx(ratio, rel=1e-3)
