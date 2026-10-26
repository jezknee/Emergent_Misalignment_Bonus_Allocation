"""Parse the replies from the scenario-choice experiment.

Run from the folder with scenario_prompts.py and scenario_functions.py:

    python parse_scenario_responses.py --responses "safe_scen_responses_*.csv" \
        --prompts scenario_prompts.csv --allocations scenario_prompts_allocations_long.csv

Writes (new files only):
  scenario_choices.csv       one row per reply: model_name, company_id, prompt_id, variant, sample_idx,
                             scenario_code (the code chosen), scenario_name, status, approach
  scenario_choices_long.csv  the same plus one row per employee, with bonus_model = the amount that
                             employee receives under the chosen scenario (pool 100,000)
  scenario_lookup.csv        scenario_code, scenario_name, and the description the model was shown
                             (one column for analyst prompts, one for assistant prompts)

status is ok / unparsed / unknown_code / not_offered. scenario_code is empty unless status is ok.
"""
import argparse
import glob
import json

import pandas as pd

from scenario_functions import SCENARIO_ORDER, SCENARIO_CODES, describe
from scenario_prompts import parse_choice

p = argparse.ArgumentParser()
p.add_argument("--responses", default="safe_scen_responses_*.csv", help="glob for the sanitised reply CSVs")
p.add_argument("--prompts", default="scenario_prompts.csv")
p.add_argument("--allocations", default="scenario_prompts_allocations_long.csv")
p.add_argument("--out-prefix", default="scenario_", help="prefix of the three output files")
args = p.parse_args()

# ---- lookup table: code, name, description as shown to the model
lookup = pd.DataFrame([{
    "scenario_code": SCENARIO_CODES[k],
    "scenario_name": k,
    "description_analyst_prompts": describe(k, True),
    "description_assistant_prompts": describe(k, False),
} for k in SCENARIO_ORDER]).sort_values("scenario_code")
lookup.to_csv(f"{args.out_prefix}lookup.csv", index=False)
code_to_name = dict(zip(lookup["scenario_code"], lookup["scenario_name"]))

# ---- replies and the prompts they answered
files = sorted(glob.glob(args.responses))
if not files:
    raise SystemExit(f"No files match {args.responses!r}")
resp = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
prompts = pd.read_csv(args.prompts)
keep = [c for c in ["prompt_id", "seed", "company_dir", "variant", "persona", "oversight", "scenario_codes"]
        if c in prompts.columns]
resp = resp.merge(prompts[keep], on="prompt_id", how="left", validate="many_to_one")
if resp["scenario_codes"].isna().any():
    raise SystemExit("Some replies have a prompt_id that is not in the prompts file.")

# safe_export puts an apostrophe in front of cells it neutralised
resp["response"] = resp["response"].where(resp.get("neutralised", False) != True,
                                          resp["response"].str.lstrip("'"))

parsed = resp.apply(lambda r: parse_choice(r["response"], json.loads(r["scenario_codes"])), axis=1)
resp[["scenario_code", "status", "approach"]] = pd.DataFrame(parsed.tolist(), index=resp.index)
resp["scenario_name"] = resp["scenario_code"].map(code_to_name)
resp["model_name"] = resp["model"].str.split("/").str[-1]
resp["company_id"] = resp["company_dir"] if resp.get("company_dir") is not None and resp["company_dir"].notna().all() else resp["seed"]

cols = ["model_name", "company_id", "prompt_id", "variant", "persona", "oversight", "sample_idx",
        "scenario_code", "scenario_name", "status", "approach"]
choices = resp[cols].sort_values(["model_name", "company_id", "prompt_id", "sample_idx"])
choices.to_csv(f"{args.out_prefix}choices.csv", index=False)

# ---- one row per employee: the amount under the chosen scenario
alloc = pd.read_csv(args.allocations)[["prompt_id", "scenario_code", "employee", "amount"]]
long = (choices[choices["status"] == "ok"]
        .merge(alloc, on=["prompt_id", "scenario_code"], how="left")
        .rename(columns={"amount": "bonus_model"}))
long.to_csv(f"{args.out_prefix}choices_long.csv", index=False)

# ---- quick report
print(f"{len(choices)} replies from {choices['model_name'].nunique()} models")
print(pd.crosstab(choices["model_name"], choices["status"], margins=True))
print("\nChoices (ok replies only):")
print(pd.crosstab(choices["scenario_code"], choices["model_name"]))
print(f"\nWrote {args.out_prefix}choices.csv, {args.out_prefix}choices_long.csv, {args.out_prefix}lookup.csv")
