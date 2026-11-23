"""Run every estimator on every sample, bootstrap the intervals and write causal/ outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data  # noqa: E402
from . import estimators as E  # noqa: E402

OUT = data.ROOT / "causal"
BALANCE_COLS = E.OLS_COLS


def run(reps: int = 500, seed: int = 20261201, out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows, balance, sens = [], {}, []
    benchmark = None
    for i, sample in enumerate(data.SAMPLES):
        df = data.load(sample)
        fit = E.estimate_all(df)
        boot = E.bootstrap(df, reps, seed + i)
        for m in E.METHODS:
            lo, hi = np.percentile(boot[m], [2.5, 97.5])
            rows.append(
                {
                    "sample": sample,
                    "method": m,
                    "estimate": fit.estimates[m],
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "boot_se": float(boot[m].std(ddof=1)),
                }
            )
        if sample == "experimental":
            benchmark = fit.estimates["diff_means"]
            welch = E.welch_ci(df)
        t = df["treat"].to_numpy() == 1
        matched = pd.concat([df[t], df.loc[fit.match_idx]], ignore_index=True)
        balance[sample] = pd.DataFrame(
            {
                "mean_treated": df.loc[t, BALANCE_COLS].mean(),
                "mean_control": df.loc[~t, BALANCE_COLS].mean(),
                "smd_raw": E.smd(df, BALANCE_COLS),
                "smd_matched": E.smd(matched, BALANCE_COLS),
                "smd_ipw": E.smd(df, BALANCE_COLS, fit.weights),
            }
        )
        balance[sample].round(3).to_csv(out / f"balance_{sample}.csv", index_label="covariate")
        diag = {
            "n_treated": int(t.sum()),
            "n_control": int((~t).sum()),
            "distinct_matched_controls": int(len(set(fit.match_idx.tolist()))),
            "ipw_effective_n_controls": E.effective_n(fit.weights[~t]),
            "share_controls_below_min_treated_ps": float((fit.ps[~t] < fit.ps[t].min()).mean()),
            "share_treated_above_max_control_ps": float((fit.ps[t] > fit.ps[~t].max()).mean()),
            "max_abs_smd_raw": float(balance[sample]["smd_raw"].abs().max()),
            "max_abs_smd_matched": float(balance[sample]["smd_matched"].abs().max()),
            "max_abs_smd_ipw": float(balance[sample]["smd_ipw"].abs().max()),
        }
        balance[sample + "_diag"] = diag
        if sample != "experimental":
            simple = E.estimate_all(df, spec="simple").estimates
            for m in ("psm", "ipw", "aipw"):
                sens.append(
                    {
                        "sample": sample,
                        "method": m,
                        "dw_spec": fit.estimates[m],
                        "linear_spec": simple[m],
                    }
                )
        overlap_plot(df, fit.ps, sample, out / f"overlap_{sample}.png")

    est = pd.DataFrame(rows)
    est["bias_vs_benchmark"] = est["estimate"] - benchmark
    est["ci_covers_benchmark"] = (est["ci_low"] <= benchmark) & (benchmark <= est["ci_high"])
    est.round(2).to_csv(out / "estimates.csv", index=False)
    pd.DataFrame(sens).round(2).to_csv(out / "ps_spec_sensitivity.csv", index=False)
    forest_plot(est, benchmark, out / "estimates_vs_benchmark.png")

    summary = {
        "benchmark": benchmark,
        "benchmark_welch_ci": list(welch),
        "bootstrap_reps": reps,
        "seed": seed,
        "diagnostics": {s: balance[s + "_diag"] for s in data.SAMPLES},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def overlap_plot(df: pd.DataFrame, ps: np.ndarray, sample: str, path: Path) -> None:
    lp = np.log(ps / (1 - ps))
    t = df["treat"].to_numpy() == 1
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(lp.min(), lp.max(), 50)
    ax.hist(lp[~t], bins=bins, density=True, alpha=0.55, label=f"controls (n={(~t).sum()})")
    ax.hist(lp[t], bins=bins, density=True, alpha=0.55, label=f"NSW treated (n={t.sum()})")
    ax.set_xlabel("logit of estimated propensity score")
    ax.set_ylabel("density")
    ax.set_title(f"Overlap: {data.SAMPLES[sample]}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def forest_plot(est: pd.DataFrame, benchmark: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels, y = [], 0
    for sample in data.SAMPLES:
        for m in E.METHODS:
            r = est[(est["sample"] == sample) & (est["method"] == m)].iloc[0]
            ax.errorbar(
                r.estimate,
                y,
                xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]],
                fmt="o",
                color="C0" if sample == "experimental" else "C1",
                capsize=3,
            )
            labels.append(f"{sample}: {E.METHODS[m]}")
            y -= 1
        y -= 0.5
    ax.axvline(benchmark, color="k", lw=1, ls="--", label="experimental benchmark")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_yticks(_ypos())
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("effect on 1978 earnings, USD (95% bootstrap CI)")
    ax.set_xlim(-18000, 7000)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _ypos() -> list[float]:
    pos, y = [], 0.0
    for _ in data.SAMPLES:
        for _ in E.METHODS:
            pos.append(y)
            y -= 1
        y -= 0.5
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    args = ap.parse_args()
    s = run(args.reps)
    print(json.dumps({"benchmark": s["benchmark"]}, indent=1))


if __name__ == "__main__":
    main()
