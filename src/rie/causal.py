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
    samples = {s: data.load(s) for s in data.SAMPLES}
    fits = {s: E.estimate_all(df) for s, df in samples.items()}
    benchmark = fits["experimental"].estimates["diff_means"]

    boot = E.bootstrap_joint(samples, reps, seed)
    wide = boot.pivot_table(index="rep", columns=["sample", "method"], values="estimate")
    bench_draws = wide[("experimental", "diff_means")]
    rows = []
    for s, fit in fits.items():
        for m in E.METHODS:
            draws = wide[(s, m)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            r = {
                "sample": s,
                "method": m,
                "estimate": fit.estimates[m],
                "ci_low": float(lo),
                "ci_high": float(hi),
                "boot_se": float(draws.std(ddof=1)),
                "ci_covers_benchmark": bool(lo <= benchmark <= hi),
            }
            if s == "experimental":
                # Same data as the benchmark: any gap is sampling noise, not bias.
                r.update(bias_vs_benchmark=np.nan, bias_ci_low=np.nan, bias_ci_high=np.nan)
            else:
                blo, bhi = np.percentile(draws - bench_draws, [2.5, 97.5])
                r.update(
                    bias_vs_benchmark=fit.estimates[m] - benchmark,
                    bias_ci_low=float(blo),
                    bias_ci_high=float(bhi),
                )
            rows.append(r)
    est = pd.DataFrame(rows)
    est["bias_ci_excludes_zero"] = (est["bias_ci_low"] > 0) | (est["bias_ci_high"] < 0)
    est.round(4).to_csv(out / "estimates.csv", index=False)

    diagnostics, sens = {}, []
    for s, df in samples.items():
        fit = fits[s]
        t = df["treat"].to_numpy() == 1
        sd_raw = E.pooled_sd(df, BALANCE_COLS)
        matched = pd.concat([df[t], df.loc[fit.match_idx]], ignore_index=True)
        bal = pd.DataFrame(
            {
                "mean_treated": df.loc[t, BALANCE_COLS].mean(),
                "mean_control": df.loc[~t, BALANCE_COLS].mean(),
                "smd_raw": E.smd(df, BALANCE_COLS),
                "smd_matched": E.smd(matched, BALANCE_COLS, sd_ref=sd_raw),
                "smd_ipw": E.smd(df, BALANCE_COLS, fit.weights, sd_ref=sd_raw),
            }
        )
        bal.round(3).to_csv(out / f"balance_{s}.csv", index_label="covariate")
        diagnostics[s] = {
            "n_treated": int(t.sum()),
            "n_control": int((~t).sum()),
            "distinct_matched_controls": int(len(set(fit.match_idx.tolist()))),
            "ipw_effective_n_controls": E.effective_n(fit.weights[~t]),
            "ipw_top5_control_weight_share": E.top_weight_share(fit.weights[~t], 5),
            "share_controls_below_min_treated_ps": float((fit.ps[~t] < fit.ps[t].min()).mean()),
            "n_treated_above_max_control_ps": int((fit.ps[t] > fit.ps[~t].max()).sum()),
            "max_abs_smd_raw": float(bal["smd_raw"].abs().max()),
            "max_abs_smd_matched": float(bal["smd_matched"].abs().max()),
            "max_abs_smd_ipw": float(bal["smd_ipw"].abs().max()),
            "covariates_smd_ipw_above_0.1": bal.index[bal["smd_ipw"].abs() > 0.1].tolist(),
        }
        if s != "experimental":
            simple = E.estimate_all(df, spec="simple").estimates
            tr_df = E.trimmed(df)
            trim = E.estimate_all(tr_df).estimates
            for m in ("psm", "ipw", "aipw"):
                sens.append(
                    {
                        "sample": s,
                        "method": m,
                        "dw_spec": fit.estimates[m],
                        "linear_spec": simple[m],
                        "trimmed_0.1_0.9": trim[m],
                        "trimmed_n_treated": int(tr_df["treat"].sum()),
                        "trimmed_n_control": int((tr_df["treat"] == 0).sum()),
                    }
                )
        overlap_plot(df, fit.ps, s, out / f"overlap_{s}.png")

    pd.DataFrame(sens).round(4).to_csv(out / "ps_spec_sensitivity.csv", index=False)
    forest_plot(est, benchmark, out / "estimates_vs_benchmark.png")
    summary = {
        "benchmark": benchmark,
        "benchmark_welch_ci": list(E.welch_ci(samples["experimental"])),
        "bootstrap_reps": reps,
        "bootstrap": "joint: treated drawn once per resample and shared by all samples",
        "seed": seed,
        "diagnostics": diagnostics,
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
                xerr=[[max(r.estimate - r.ci_low, 0)], [max(r.ci_high - r.estimate, 0)]],
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
    pad = 0.05 * (est["ci_high"].max() - est["ci_low"].min())
    ax.set_xlim(est["ci_low"].min() - pad, est["ci_high"].max() + pad)
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
