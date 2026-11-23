"""Tests for the hypothetical pilot design. All data here is simulated."""

import json

import numpy as np
import pandas as pd
import pytest

from rie import pilot as P
from rie.data import ROOT


def test_n_required_pinned_to_hand_computed_value():
    # 2 * ceil(((1.95996 * sqrt(2 * .825 * .175) + 0.84162 * sqrt(.16 + .1275)) / .05) ** 2 / 2)
    assert P.n_required(0.05, 0.80) == 1812


def test_mde_inverts_n_required():
    n = P.n_required(0.05, 0.80)
    assert P.mde_individual(n, 0.80) == pytest.approx(0.05, abs=5e-4)


def test_cluster_mde_grows_with_icc_and_exceeds_individual():
    t = P.power_table()
    assert t["mde_individual_pp"].is_monotonic_decreasing
    cols = [c for c in t.columns if c.startswith("mde_cluster")]
    for c in cols:
        assert (t[c] > t["mde_individual_pp"]).all()
    assert (t[cols].diff(axis=1).iloc[:, 1:] > 0).all().all()


def test_cluster_mde_at_icc_zero_differs_only_by_t_quantiles():
    ind = P.mde_individual(1900, 0.8)
    clu0 = P.mde_cluster(1900, 12, 0.8, 0.0)
    many = P.mde_cluster(1900, 1900, 0.8, 0.0)
    assert clu0 > ind
    assert many == pytest.approx(ind, abs=2e-4)


def test_stratified_assignment_is_balanced_within_strata():
    rng = np.random.default_rng(3)
    strata = pd.Series(rng.choice(list("abcde"), 501))
    t = P.stratified_assignment(strata, rng)
    for _, g in pd.Series(t).groupby(strata):
        assert abs(g.sum() - (len(g) - g.sum())) <= 1


def test_calibration_hits_planning_targets_on_heterogeneous_hazards():
    rel = np.exp(np.random.default_rng(8).normal(0, 0.6, 5000))
    h0, hr = P.calibrate(rel, P.A)
    assert ((1 - h0 * rel) ** 12).mean() == pytest.approx(0.80, abs=1e-6)
    assert ((1 - h0 * hr * rel) ** 12).mean() == pytest.approx(0.85, abs=1e-6)


def test_simulation_is_labelled_and_reproducible():
    a, _ = P.simulate(300, seed=7)
    b, _ = P.simulate(300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert (a["simulated"] == 1).all()
    assert set(a["months"]) <= set(range(1, 25))
    assert (a["left_m13_15"] == a["retained_12m"] - a["retained_15m"]).all()


def test_planned_analysis_recovers_effect_on_large_simulation():
    df, _ = P.simulate(20000, seed=11)
    r = P.primary_analysis(df)
    assert r["ci_low_pp"] < 5 < r["ci_high_pp"]
    assert P.harm_analysis(df)["ci_low_pp"] > 0  # the built-in spike is detected
    ch = P.cox_models(df)["change_at_12"]
    assert ch["ratio_ci_low"] > 1


def test_harm_check_false_alarm_rate_without_spike():
    """Harm test on everyone randomised, no spike: rejections near alpha (40 pilots)."""
    a = P.Assumptions(post_payment_hazard_mult=1.0)
    rejections = sum(
        P.harm_analysis(P.simulate(1900, 500 + r, a)[0])["p_value"] < 0.05 for r in range(40)
    )
    assert rejections <= 6  # P(X >= 7) < 0.02 if the true rate is 5%


def test_no_effect_world_gives_null_results():
    a = P.Assumptions(target_effect_pp=1e-6, post_payment_hazard_mult=1.0)
    df, _ = P.simulate(6000, seed=5, a=a)
    r = P.primary_analysis(df)
    assert r["ci_low_pp"] < 0 < r["ci_high_pp"]


def test_committed_pilot_results_reproduce():
    committed = json.loads((ROOT / "design" / "pilot_results.json").read_text())
    df, _ = P.simulate(committed["n_pilot"], committed["seed"])
    assert P.primary_analysis(df)["effect_pp"] == pytest.approx(
        committed["primary"]["effect_pp"], abs=1e-9
    )
    assert P.harm_analysis(df)["effect_pp"] == pytest.approx(
        committed["harm_m13_15"]["effect_pp"], abs=1e-9
    )


def test_budget_arithmetic_in_results():
    c = json.loads((ROOT / "design" / "pilot_results.json").read_text())
    a = c["assumptions"]
    paid = c["cost"]["planned_n_paid"]
    assert c["cost"]["planned_budget_dkk"] == pytest.approx(paid * a["bonus_dkk"])
