import re
import pandas as pd
from load_companies import iter_companies
from bonus_functions import get_all_bonuses


def get_bonuses_for_all_companies(root="companies", pool=100000, current_only=False):
    """One dataframe: a row per employee per company, with company number and name."""
    frames = []
    for folder, company_tables, metadata in iter_companies(root):
        bonuses = get_all_bonuses(company_tables, pool=pool, current_only=current_only)
        bonuses.insert(0, "company", folder)
        bonuses.insert(0, "company_number", int(re.search(r"\d+", folder).group()))
        frames.append(bonuses)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    all_bonuses = get_bonuses_for_all_companies("companies")
    all_bonuses.to_csv("all_bonuses.csv", index=False)
    print(all_bonuses.shape)
    print(all_bonuses.head())
