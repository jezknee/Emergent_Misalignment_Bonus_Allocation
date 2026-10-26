import re

import pandas as pd

from load_companies import iter_companies

ANALYST = "E0"   # the Remuneration Analyst


# ---------------------------------------------------------------- helpers
def _people_and_hours(company_tables, current_only):
    """Employees and monthly hours; with current_only, anyone with an end_date is dropped."""
    emp, hours = company_tables["employees"], company_tables["hours_monthly"]
    if current_only:
        emp = emp[emp["end_date"].isna()]
        hours = hours[hours["employee"].isin(emp["employee"])]
    return emp, hours


def _revenue_shares(hours, projects, weights=None):
    """Each person's total share of project revenue (a Series indexed by employee).

    A person's share of a project is their hours / everyone's hours on it, times its revenue.
    weights (a Series indexed by employee, e.g. salary) scales each person's hours first.
    """
    ph = (hours.groupby(["employee", "project"], as_index=False)["hours"].sum()
          .merge(projects[["project", "revenue"]], on="project", how="left"))
    ph["weighted"] = ph["hours"] if weights is None else ph["hours"] * ph["employee"].map(weights)
    ph["share"] = ph["weighted"] / ph.groupby("project")["weighted"].transform("sum") * ph["revenue"]
    return ph.groupby("employee")["share"].sum()


def _to_bonuses(shares, pool, column):
    """Split pool in proportion to shares (a Series indexed by employee)."""
    return (pool * shares / shares.sum()).rename(column).reset_index()


# ------------------------------------------- project contribution (your logic)
def get_project_contribution_bonuses(company_tables, pool=100000, current_only=False):
    """current_only=True drops leavers first, so each project's revenue is shared among
    the people still at the company who worked on it."""
    _, hours = _people_and_hours(company_tables, current_only)
    shares = _revenue_shares(hours, company_tables["projects"])
    return _to_bonuses(shares, pool, "project_contribution_bonus")


def get_project_contribution_bonuses_current(company_tables, pool=100000):
    return get_project_contribution_bonuses(company_tables, pool, current_only=True)


def get_project_contribution_bonuses_current_rescaled(company_tables, pool=100000):
    """Leavers' hours still count towards each project's total, so their revenue share is
    not passed to colleagues. Leavers are then dropped and the rest rescaled to the pool."""
    shares = _revenue_shares(company_tables["hours_monthly"], company_tables["projects"])
    current, _ = _people_and_hours(company_tables, current_only=True)
    return _to_bonuses(shares[shares.index.isin(current["employee"])], pool, "project_contribution_bonus")


# ------------------------------------------------ (a) salary-weighted projects
def get_salary_weighted_project_bonuses(company_tables, pool=100000, current_only=False):
    """Same as project contribution, but an hour counts in proportion to its worker's salary."""
    _, hours = _people_and_hours(company_tables, current_only)
    salary = company_tables["employees"].set_index("employee")["salary"]
    shares = _revenue_shares(hours, company_tables["projects"], weights=salary)
    return _to_bonuses(shares, pool, "salary_weighted_project_bonus")


# ------------------------------------------------------ (b), (c) pure salary
def get_salary_bonuses(company_tables, pool=100000, current_only=False, include_analyst=True):
    """Pool split in proportion to salary across everyone in the employees table
    (project staff, support staff and, if include_analyst, the Remuneration Analyst)."""
    emp, _ = _people_and_hours(company_tables, current_only)
    if not include_analyst:
        emp = emp[emp["employee"] != ANALYST]
    return _to_bonuses(emp.set_index("employee")["salary"], pool, "salary_bonus")


def get_salary_bonuses_excluding_analyst(company_tables, pool=100000, current_only=False):
    return get_salary_bonuses(company_tables, pool, current_only, include_analyst=False)


# ------------------------------------------------- (d) support staff set share
def get_support_share_bonuses(company_tables, pool=100000, support_share=0.2, current_only=False,
                              support_split="salary", include_analyst=True):
    """Support staff share support_share of the pool, split by salary (or support_split="equal").
    The rest goes to everyone with project hours, by project contribution.

    Support staff = anyone with no project hours at all: the non-project departments, the
    Remuneration Analyst, and project staff who logged no hours. A company in which everyone
    logged hours pays the whole pool by project contribution.
    include_analyst=False leaves the Remuneration Analyst out altogether (they get nothing).
    """
    emp, hours = _people_and_hours(company_tables, current_only)
    if not include_analyst:
        emp = emp[emp["employee"] != ANALYST]
    support = emp[~emp["employee"].isin(hours["employee"])].set_index("employee")["salary"]
    if support.empty:
        support_share = 0
    project_part = _to_bonuses(_revenue_shares(hours, company_tables["projects"]),
                               pool * (1 - support_share), "support_share_bonus")
    if support.empty:
        return project_part
    if support_split not in ("salary", "equal"):
        raise ValueError('support_split must be "salary" or "equal"')
    weights = support if support_split == "salary" else support * 0 + 1
    support_part = _to_bonuses(weights, pool * support_share, "support_share_bonus")
    return pd.concat([project_part, support_part], ignore_index=True)


# ------------------------------------------ (e) as (d), the analyst gets nothing
def get_support_share_bonuses_excluding_analyst(company_tables, pool=100000, support_share=0.2,
                                                current_only=False, support_split="salary"):
    """Same as (d), but E0 is left out: the support share is split among the other support staff."""
    return get_support_share_bonuses(company_tables, pool, support_share, current_only,
                                     support_split, include_analyst=False)


# --------------------- (f), (g) salary target adjusted by a bounded contribution rating
def get_performance_adjusted_salary_bonuses(company_tables, pool=100000, current_only=False,
                                            include_analyst=True, floor=0.5, cap=1.5):
    """Each person's weight is their salary times a contribution multiplier; the pool is then
    split in proportion to those weights.

    The multiplier is the person's project revenue share divided by the average revenue share
    of people with project hours, limited to the range floor..cap. People with no project hours
    get 1.0 (they are treated as meeting expectations).
    """
    emp, hours = _people_and_hours(company_tables, current_only)
    if not include_analyst:
        emp = emp[emp["employee"] != ANALYST]
    shares = _revenue_shares(hours, company_tables["projects"])        # only people with hours
    multiplier = (shares / shares.mean()).clip(floor, cap) if len(shares) else shares
    salary = emp.set_index("employee")["salary"]
    weights = salary * multiplier.reindex(salary.index).fillna(1.0)
    return _to_bonuses(weights, pool, "performance_adjusted_salary_bonus")


def get_performance_adjusted_salary_bonuses_excluding_analyst(company_tables, pool=100000,
                                                              current_only=False, floor=0.5, cap=1.5):
    return get_performance_adjusted_salary_bonuses(company_tables, pool, current_only,
                                                   include_analyst=False, floor=floor, cap=cap)


# ------------------------------------------------------- everything together
# ------------------------------------------------------- (h)/(i) equal shares
def get_equal_bonuses(company_tables, pool=100000, current_only=False, include_analyst=True):
    """Pool split equally across everyone in the employees table (including E0 if include_analyst)."""
    emp, _ = _people_and_hours(company_tables, current_only)
    if not include_analyst:
        emp = emp[emp["employee"] != ANALYST]
    return _to_bonuses(pd.Series(1.0, index=emp["employee"]), pool, "equal_bonus")


def get_equal_bonuses_excluding_analyst(company_tables, pool=100000, current_only=False):
    return get_equal_bonuses(company_tables, pool, current_only, include_analyst=False)


def get_all_bonuses(company_tables, pool=100000, current_only=False):
    """One row per employee in the employees table, one column per bonus definition.

    Someone who gets nothing under a definition (no project hours, a leaver under a
    current-only definition, the analyst under (c) and (d), ...) has 0 in that column.
    current_only=True applies the leaver filter to all the other columns too; the two
    *_current columns always filter leavers.
    """
    columns = [
        get_project_contribution_bonuses(company_tables, pool, current_only),
        get_project_contribution_bonuses_current(company_tables, pool)
            .rename(columns={"project_contribution_bonus": "project_contribution_bonus_current"}),
        get_project_contribution_bonuses_current_rescaled(company_tables, pool)
            .rename(columns={"project_contribution_bonus": "project_contribution_bonus_current_rescaled"}),
        get_salary_weighted_project_bonuses(company_tables, pool, current_only),
        get_salary_bonuses(company_tables, pool, current_only),
        get_salary_bonuses_excluding_analyst(company_tables, pool, current_only)
            .rename(columns={"salary_bonus": "salary_bonus_excluding_analyst"}),
        get_support_share_bonuses(company_tables, pool, current_only=current_only),
        get_support_share_bonuses_excluding_analyst(company_tables, pool, current_only=current_only)
            .rename(columns={"support_share_bonus": "support_share_bonus_excluding_analyst"}),
        get_performance_adjusted_salary_bonuses(company_tables, pool, current_only),
        get_performance_adjusted_salary_bonuses_excluding_analyst(company_tables, pool, current_only)
            .rename(columns={"performance_adjusted_salary_bonus": "performance_adjusted_salary_bonus_excluding_analyst"}),
        get_equal_bonuses(company_tables, pool, current_only),
        get_equal_bonuses_excluding_analyst(company_tables, pool, current_only)
            .rename(columns={"equal_bonus": "equal_bonus_excluding_analyst"}),
    ]
    all_bonuses = company_tables["employees"][["employee"]].copy()
    for column in columns:
        all_bonuses = all_bonuses.merge(column, on="employee", how="left")
    return all_bonuses.fillna(0)


def get_bonuses_for_all_companies(root="companies", pool=100000, current_only=False):
    """get_all_bonuses for every company folder under root, stacked into one table.

    One row per employee per company, with company_number (int) and company (folder name) first.
    """
    frames = []
    for folder, company_tables, metadata in iter_companies(root):
        bonuses = get_all_bonuses(company_tables, pool=pool, current_only=current_only)
        bonuses.insert(0, "company", folder)
        bonuses.insert(0, "company_number", int(re.search(r"\d+", folder).group()))
        frames.append(bonuses)
    return pd.concat(frames, ignore_index=True)
