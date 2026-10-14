"""Check that a responses file from run_allocation.py is complete and sane.

Usage:  python check_responses.py responses_<model>.jsonl --prompts prompts.jsonl --n-samples 5

Prints a short report and exits with a non-zero status if anything is missing.
Run it on the pod before downloading, and again locally afterwards. It only counts
and inspects text: it never runs or evaluates anything in a response.
"""
import argparse
import json
import sys
from collections import Counter

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("responses")
    p.add_argument("--prompts", required=True, help="prompts.jsonl used for the run")
    p.add_argument("--n-samples", type=int, required=True, help="samples per prompt you asked for")
    p.add_argument("--variants", help="comma-separated variants, if the run used --variants")
    p.add_argument("--limit", type=int, help="the --limit used for the run, if any")
    a = p.parse_args()

    prompts = pd.read_json(a.prompts, lines=True, dtype=False, convert_dates=False)
    if a.variants:
        prompts = prompts[prompts["variant"].isin(a.variants.split(","))]
    if a.limit:
        prompts = prompts.head(a.limit)
    expected_ids = set(prompts["prompt_id"])

    rows, bad_lines = [], 0
    with open(a.responses, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    df = pd.DataFrame(rows)
    problems = []

    def report(label, value, bad=False):
        print(f"{'FAIL' if bad else 'ok  '}  {label}: {value}")
        if bad:
            problems.append(label)

    report("lines that are not valid JSON", bad_lines, bad_lines > 0)
    if df.empty:
        report("rows", 0, True)
        sys.exit(1)
    report("models in file", sorted(df["model"].unique()), df["model"].nunique() != 1)
    n_expected = len(expected_ids) * a.n_samples
    report("rows (expected %d)" % n_expected, len(df), len(df) != n_expected)
    dup = int(df.duplicated(["prompt_id", "sample_idx"]).sum())
    report("duplicate (prompt, sample) rows", dup, dup > 0)
    missing = expected_ids - set(df["prompt_id"])
    extra = set(df["prompt_id"]) - expected_ids
    report("prompts with no responses", len(missing), len(missing) > 0)
    report("responses for prompts not in the table", len(extra), len(extra) > 0)
    per = df.groupby("prompt_id")["sample_idx"].nunique()
    short = int((per < a.n_samples).sum())
    report("prompts with fewer than %d samples" % a.n_samples, short, short > 0)
    empty = int(df["response"].fillna("").str.strip().eq("").sum())
    report("empty responses", empty, empty > 0.05 * len(df))
    fin = Counter(df["finish_reason"])
    cut = fin.get("length", 0)
    report("finish reasons", dict(fin), cut > 0.05 * len(df))
    print("      (replies cut off at --max-tokens are likely incomplete JSON; raise --max-tokens if many)")

    def parses(text):
        try:
            return isinstance(json.loads(text.strip()), dict)
        except Exception:
            return False
    ok = df["response"].fillna("").map(parses).mean()
    print(f"info  responses that parse as a JSON object: {100 * ok:.1f}% (a pipeline sanity check, not scoring)")
    print(f"info  characters per response: median {int(df['response'].str.len().median())}, "
          f"max {int(df['response'].str.len().max())}")
    print("\nRESULT:", "PROBLEMS: " + ", ".join(problems) if problems else "all checks passed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
