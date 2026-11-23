"""Tests for the hypothetical pilot design. All data here is simulated."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from rie import pilot as P


def test_n_required_close_to_statsmodels_arcsine_formula():
    ours = P.n_required(0.05, 0.80)
    h = proportion_effectsize(0.85, 0.80)
    sm_arm = NormalIndPower().solve_power(abs(h), alpha=0.05, power=0.8)
    assert ours == pytest.approx(2 * sm_arm, rel=0.05)


def test_mde_inverts_n_required():
    n = P.n_required(0.05, 0.80)
    assert P.mde_individual(n, 0.80) == pytest.approx(0.05, abs=5e-4)


def test_mde_shrinks_with_n_and_cluster_design_is_worse():
    t = P.power_table()
    assert t["mde_individual_pp"].is_monotonic_decreasing
    assert (t["mde_cluster_pp"] > t["mde_individual_pp"]).all()


def test_stratified_assignment_is_balanced_within_strata():
    rng = np.random.default_rng(3)
    strata = pd.Series(rng.choice(list("abcde"), 501))
    t = P.stratified_assignment(strata, rng)
    for _, g in pd.Series(t).groupby(strata):
        assert abs(g.sum() - (len(g) - g.sum())) <= 1


def test_calibration_hits_planning_targets():
    df, truth = P.simulate(500, seed=1)
    rel = np.ones(10)
    h0, hr = P.calibrate(rel, P.A)
    assert (1 - h0) ** 12 == pytest.approx(0.80, abs=1e-6)
    assert (1 - h0 * hr) ** 12 == pytest.approx(0.85, abs=1e-6)
    assert 0 < truth["hazard_ratio_months_1_12"] < 1


def test_simulation_is_labelled_and_reproducible():
    a, _ = P.simulate(300, seed=7)
    b, _ = P.simulate(300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert (a["simulated"] == 1).all()
    assert set(a["months"]) <= set(range(1, 25))


def test_planned_analysis_recovers_effect_on_large_simulation():
    df, _ = P.simulate(20000, seed=11)
    r = P.primary_analysis(df)
    assert r["ci_low_pp"] < 5 < r["ci_high_pp"]
    harm = P.side_effect_analysis(df)
    assert harm["ci_low_pp"] > 0  # the built-in post-payment spike is detected
    cx = P.cox_models(df)
    assert cx["months_0_12"]["hazard_ratio"] < 1 < cx["months_13_24"]["hazard_ratio"]


def test_no_effect_world_gives_null_results():
    a = P.Assumptions(target_effect_pp=1e-6, post_payment_hazard_mult=1.0)
    df, _ = P.simulate(6000, seed=5, a=a)
    r = P.primary_analysis(df)
    assert r["ci_low_pp"] < 0 < r["ci_high_pp"]
