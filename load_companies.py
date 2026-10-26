"""Loop over the exported company folders.

    from load_companies import iter_companies
    for name, t, meta in iter_companies("companies"):
        emp = t["employees"]            # pandas DataFrames: employees, projects,
        hours = t["hours_monthly"]      # project_details, hours_monthly, hours_total
        pool = meta["pool"]             # plain dict from company.json (seed, company_name, pool, ...)

E0 (the Remuneration Analyst) and the ground-truth columns are included; drop them yourself.
A blank end_date is loaded as missing (NaN), meaning still employed.
"""
import json
from pathlib import Path

import pandas as pd

TABLES = ["employees", "projects", "project_details", "hours_monthly", "hours_total"]


def iter_companies(root="companies"):
    for folder in sorted(Path(root).glob("company_*")):
        t = {name: pd.read_csv(folder / f"{name}.csv") for name in TABLES}
        meta = json.loads((folder / "company.json").read_text(encoding="utf-8"))
        yield folder.name, t, meta
