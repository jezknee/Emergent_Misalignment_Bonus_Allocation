"""Check YOUR exported companies and YOUR prompt table before you deploy a pod.

Usage (local machine, in the folder with the scripts):

    python check_my_data.py --companies companies --prompts prompts.jsonl

What it does (all read-only; nothing is changed or run):
  1. Every company folder (company_0000/, ...) goes through the data checks: hours totals,
     hours only inside project dates and employment dates, monthly cap, salaries and
     titles, the Remuneration Analyst, support staff, ID integrity, and so on.
  2. Every row of prompts.jsonl is compared with its company folder: the four tables
     printed in the prompt must match the CSV files value for value (with the analyst
     removed for assistant prompts), the approval sentence must match the route, the
     IDs must agree between the table, the rules and the answer template, ground-truth
     columns must be absent, and the table's metadata must match the prompt text.

Exit status 0 = everything passed, 1 = problems found (listed). The pytest files check
the SCRIPTS; this checks your actual FILES, so run both: pytest first, then this after
you have built the prompts, then deploy.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import data_checks as dc


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--companies", required=True, help="folder holding company_0000/, company_0001/, ...")
    p.add_argument("--prompts", required=True, help="prompts.jsonl (or .csv) built from those folders")
    p.add_argument("--show", type=int, default=8, help="how many problems to print per section")
    a = p.parse_args()

    root = Path(a.companies)
    problems = 0

    def section(title, found, ok_message):
        nonlocal problems
        if found:
            problems += len(found)
            print(f"FAIL  {title}: {len(found)} problem(s)")
            for line in found[:a.show]:
                print(f"        - {line}")
            if len(found) > a.show:
                print(f"        ... and {len(found) - a.show} more (use --show N to see more)")
        else:
            print(f"ok    {ok_message}")

    folders = sorted(f for f in root.glob("company_*") if f.is_dir())
    if not folders:
        print(f"FAIL  no company_* folders found under {root}")
        sys.exit(1)
    found = []
    for folder in folders:
        found += [f"{folder.name}: {m}" for m in dc.check_company_folder(folder)]
    section("company data", found, f"{len(folders)} company folders pass every data check")

    path = Path(a.prompts)
    if path.suffix == ".jsonl":
        table = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    else:
        table = pd.read_csv(path, keep_default_na=False)
    found = dc.check_prompt_table(table, root)
    section("prompts", found,
            f"{len(table)} prompts agree with their company files, the approval routes and the rules")

    if "company_dir" in table.columns:
        unused = sorted({f.name for f in folders} - set(table["company_dir"]))
        if unused:
            print(f"info  {len(unused)} company folder(s) have no prompts: {', '.join(unused[:5])}")
        print(f"info  {table['company_dir'].nunique()} companies, {len(table)} prompts, "
              f"variants: {', '.join(sorted(table['variant'].unique()))}")
        print(f"info  prompt length: median {int(table['n_chars'].median())} characters, "
              f"longest {int(table['n_chars'].max())} (about {int(table['n_chars'].max() / 3)} tokens)")
    print("\nRESULT:", f"{problems} PROBLEM(S) FOUND" if problems else "all checks passed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
