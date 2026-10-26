"""Compare consistency scores between models (company-level, paired).

Usage:  python test_consistency_scores.py consistency_scores.csv
        python test_consistency_scores.py consistency_scores.csv --out results_scores

Input: consistency_scores.csv, one row per model / company / prompt / sample, with a score
(0-1) for each reference and a best_score column.

Outputs (printed, and saved as <out>_pairwise.csv and <out>_per_reference.csv):
  * mean and median score per model;
  * Friedman test across the models (when there are 3 or more);
  * pairwise paired Wilcoxon and t-tests with Holm correction, mean difference and a
    bootstrap 95% interval over companies;
  * the same pairwise tests for each reference (exploratory), Holm-corrected across all of them;
  * a mixed model for best_score with oversight as a covariate (if statsmodels is installed).
"""
import argparse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

INDEX = ["model_name", "company_id", "prompt_id", "sample_idx"]


def company_table(df, column):
    """One row per company, one column per model: median of samples, mean over prompts."""
    return (df.groupby(INDEX)[column].first()
              .groupby(level=["model_name", "company_id", "prompt_id"]).median()
              .groupby(level=["model_name", "company_id"]).mean()
              .unstack(0))


def safe_wilcoxon(d):
    d = d.dropna()
    if (d == 0).all():
        return np.nan
    return stats.wilcoxon(d).pvalue


def bootstrap_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = d.dropna().to_numpy()
    means = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def pairwise(table, label):
    rows = []
    for a, b in combinations(table.columns, 2):
        d = (table[a] - table[b]).dropna()
        lo, hi = bootstrap_ci(d)
        rows.append({"outcome": label, "a": a, "b": b, "n_companies": len(d),
                     "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi,
                     "median_diff": d.median(),
                     "p_wilcoxon": safe_wilcoxon(d),
                     "p_ttest": stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else np.nan})
    return pd.DataFrame(rows)


def add_holm(res, column="p_wilcoxon"):
    p = res[column]
    ok = p.notna()
    res["p_holm"] = np.nan
    if ok.any():
        res.loc[ok, "p_holm"] = multipletests(p[ok], method="holm")[1]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results_scores")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    missing = [c for c in INDEX + ["best_score"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    references = [c for c in df.columns if c not in INDEX + ["best_score"]]
    models = sorted(df["model_name"].unique())
    print(f"{len(df)} responses, {df['company_id'].nunique()} companies, models: {models}\n")

    # ---- primary outcome: best_score
    best = company_table(df, "best_score").dropna()
    print("best_score per model (company level):")
    print(best.agg(["mean", "median", "std"]).round(3).to_string(), "\n")

    if best.shape[1] >= 3:
        stat, p = stats.friedmanchisquare(*[best[c] for c in best.columns])
        print(f"Friedman test across models: chi2 = {stat:.2f}, p = {p:.4g}\n")

    res = add_holm(pairwise(best, "best_score"))
    print("Pairwise comparisons of best_score (a minus b; Holm over the pairs):")
    print(res.drop(columns="outcome").round(4).to_string(index=False), "\n")
    res.to_csv(f"{a.out}_pairwise.csv", index=False)

    # ---- exploratory: each reference separately
    per_ref = []
    for ref in references:
        tab = company_table(df, ref).dropna(how="all")
        if tab.shape[1] >= 2:
            per_ref.append(pairwise(tab, ref))
    if per_ref:
        per_ref = add_holm(pd.concat(per_ref, ignore_index=True))
        print("Per-reference comparisons (exploratory; Holm over all of them):")
        print(per_ref.drop(columns=["n_companies"]).round(4).to_string(index=False), "\n")
        per_ref.to_csv(f"{a.out}_per_reference.csv", index=False)

    # ---- mixed model with oversight as a covariate
    try:
        import statsmodels.formula.api as smf
        pp = (df.groupby(INDEX)["best_score"].first()
                .groupby(level=["model_name", "company_id", "prompt_id"]).median().reset_index())
        parts = pp["prompt_id"].str.split("_")
        pp["oversight"] = parts.str[3]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = smf.mixedlm("best_score ~ C(model_name) + C(oversight)", pp,
                               groups=pp["company_id"]).fit(reml=False, method="lbfgs")
            null = smf.mixedlm("best_score ~ C(oversight)", pp,
                               groups=pp["company_id"]).fit(reml=False, method="lbfgs")
            inter = smf.mixedlm("best_score ~ C(model_name) * C(oversight)", pp,
                                groups=pp["company_id"]).fit(reml=False, method="lbfgs")
        k = len(models) - 1
        lr = 2 * (full.llf - null.llf)
        print(f"Mixed model, effect of model: chi2 = {lr:.2f} on {k} df, p = {stats.chi2.sf(lr, k):.4g}")
        k2 = k * (pp["oversight"].nunique() - 1)
        lr2 = 2 * (inter.llf - full.llf)
        print(f"Mixed model, model x oversight interaction: chi2 = {lr2:.2f} on {k2} df, "
              f"p = {stats.chi2.sf(lr2, k2):.4g}")
    except Exception as e:
        print("Mixed model skipped:", e)


if __name__ == "__main__":
    main()
