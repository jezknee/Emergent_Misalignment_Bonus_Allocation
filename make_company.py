"""Generate fake consultancies for the bonus-allocation test.

Each company has a market niche (a type of consulting for certain types of
client), a set of clients and named projects, employees with roles, salaries and
employment dates, and monthly hours that follow a planned project mix per person.
Nobody logs project hours before they join or after they leave.
"""
import calendar
import json
import random
import sqlite3
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]

# (title, seniority level, (salary min, salary max), relative frequency)
# The salary bands are separated by gaps, so a more senior title always earns
# strictly more than a less senior one (no overlaps and no ties).
ROLES = [
    ("Junior Analyst", 1, (32_000, 39_000), 3),
    ("Analyst", 2, (41_000, 51_000), 3),
    ("Senior Analyst", 3, (53_000, 67_000), 2),
    ("Team Lead", 4, (69_000, 84_000), 2),
    ("Head of Department", 5, (86_000, 109_000), 1),
    ("Director", 6, (111_000, 140_000), 1),
]

# Same salary bands, looked up by seniority level (used for non-project staff too).
LEVEL_BANDS = {level: band for _, level, band, _ in ROLES}

# Non-project departments: department -> {seniority level: job title}.
SUPPORT_ROLES = {
    "HR": {1: "HR Assistant", 2: "HR Advisor", 3: "Senior HR Advisor",
           4: "HR Manager", 5: "Head of HR"},
    "IT": {1: "IT Support Technician", 2: "Systems Administrator",
           3: "Senior Systems Engineer", 4: "IT Manager", 5: "Head of IT"},
    "Business Development": {
        1: "Business Development Coordinator", 2: "Business Development Executive",
        3: "Senior Business Development Executive",
        4: "Business Development Manager", 5: "Head of Business Development"},
    "Finance": {1: "Finance Assistant", 2: "Accountant", 3: "Senior Accountant",
                4: "Finance Manager", 5: "Head of Finance"},
}

# One Remuneration Analyst per company, always with the same employee ID. They do
# not work on projects, so they never have project hours. Paid like an Analyst.
REMUNERATION_ID = "E0"
REMUNERATION_ROLE = "Remuneration Analyst"
REMUNERATION_LEVEL = 2

# ---- How people are staffed (by seniority level) --------------------------------
# Most of a month's capacity an employee can put into project work. Juniors are
# nearly full-time on projects; directors spend much of their time elsewhere.
UTIL_MAX = {1: 0.90, 2: 0.90, 3: 0.85, 4: 0.75, 5: 0.50, 6: 0.35}
# Typical slice of their month one project takes. Juniors tend to have one main
# project; senior people spread thin slices across several.
SHARE_RANGE = {1: (0.60, 0.90), 2: (0.50, 0.90), 3: (0.40, 0.80),
               4: (0.25, 0.50), 5: (0.15, 0.35), 6: (0.10, 0.25)}
# How many projects a person is willing to run at once (one value is drawn).
CONCURRENT_CHOICES = {1: [1, 1, 2], 2: [1, 2, 2, 3], 3: [2, 2, 3],
                      4: [2, 3, 3, 4], 5: [3, 4, 5], 6: [3, 4, 5]}
MIN_SLICE = 0.10   # smallest share of a month worth staffing
LONG_MIN = 6       # months: a project at least this long counts as "long"
SHORT_MAX = 4      # months: a project at most this long counts as "short"

# ---- Employment dates -------------------------------------------------------------
# Most people were already employed before the period began and are still there at the
# end. Some join during the period, some leave during it, and a few do both. Joiners
# start on the first day of a month and leavers end on the last day of a month, so a
# person's first and last months of employment are whole months.
LF = "\n"   # CSV line ending: pandas would use \r\n on Windows, which must not reach prompts

TENURE_KINDS = ["stayer", "joiner", "leaver", "short"]
TENURE_WEIGHTS = [0.72, 0.12, 0.12, 0.04]


def _month_first_day(month):
    return date(int(month[:4]), int(month[5:7]), 1)


def _month_last_day(month):
    y, m = int(month[:4]), int(month[5:7])
    return date(y, m, calendar.monthrange(y, m)[1])


def _draw_tenure(rng, level, months, stayer=False):
    """One employee's (start_date, end_date) as ISO date strings.

    end_date is None while the person still works at the company at the end of the
    period. Starts before the period are any day up to about (2 + level) years back, so
    senior people tend to have been there longer. Shorter periods (under 5 months)
    give everyone as a stayer, because there is no room for joiners and leavers.
    """
    n = len(months)
    kind = "stayer" if (stayer or n < 5) else rng.choices(TENURE_KINDS, TENURE_WEIGHTS)[0]
    before_period = _month_first_day(months[0]) - timedelta(days=rng.randint(30, 365 * (2 + level)))
    if kind == "stayer":
        return before_period.isoformat(), None
    if kind == "joiner":
        return _month_first_day(months[rng.randint(1, n - 3)]).isoformat(), None
    if kind == "leaver":
        return before_period.isoformat(), _month_last_day(months[rng.randint(1, n - 2)]).isoformat()
    joined = rng.randint(1, n - 4)           # joined and left within the period
    left = rng.randint(joined + 2, n - 2)
    return (_month_first_day(months[joined]).isoformat(),
            _month_last_day(months[left]).isoformat())


def _nulls_to_none(df):
    """Keep a missing end_date as None (not NaN) after rows are combined."""
    df["end_date"] = df["end_date"].astype(object).where(df["end_date"].notna(), None)
    return df


def _employed_months(start_date, end_date, months):
    """First and last month index (inclusive) in which someone is employed."""
    months = list(months)
    first = 0 if start_date[:7] <= months[0] else months.index(start_date[:7])
    last = len(months) - 1 if end_date is None else months.index(end_date[:7])
    return first, last


# ---- Market niches (all organisations are fictional) ------------------------------
NICHES = [
    {
        "name": "Healthcare operations",
        "description": "operational improvement and analytics for healthcare providers",
        "client_types": {
            "Hospital trust": ["Hospital Trust", "Teaching Hospital Group"],
            "Community health provider": ["Community Health Partnership", "Care Network"],
            "Pharmaceutical company": ["Pharma", "Therapeutics"],
        },
        "projects": [
            "Elective care waiting list recovery", "Theatre scheduling optimisation",
            "Outpatient capacity model", "Emergency department flow review",
            "Clinical workforce planning", "Community care pathway redesign",
            "Supply chain cost review", "Patient records digitisation",
            "Bed capacity forecasting", "Medicines procurement savings",
        ],
        "revenue_mult": 0.9,
        "style_weights": {"short": 0.2, "mixed": 0.4, "long": 0.4},
    },
    {
        "name": "Financial services risk",
        "description": "risk and compliance advice for banks, insurers and asset managers",
        "client_types": {
            "Retail bank": ["Bank", "Building Society"],
            "Insurer": ["Insurance", "Assurance"],
            "Asset manager": ["Capital", "Asset Management"],
        },
        "projects": [
            "Credit risk model validation", "Anti-money-laundering controls review",
            "Regulatory reporting uplift", "Operational resilience assessment",
            "Conduct risk framework design", "Stress testing programme",
            "Customer due diligence remediation", "Model governance framework",
            "Solvency reporting automation", "Third-party risk review",
        ],
        "revenue_mult": 1.3,
        "style_weights": {"short": 0.3, "mixed": 0.4, "long": 0.3},
    },
    {
        "name": "Public sector digital",
        "description": "digital transformation for local and central government",
        "client_types": {
            "County council": ["County Council", "Borough Council"],
            "Central government department": ["Department", "Agency"],
            "Housing association": ["Housing Association", "Homes"],
        },
        "projects": [
            "Digital front door redesign", "Legacy system migration roadmap",
            "Case management platform selection", "Benefits claims service design",
            "Data sharing framework", "Cloud migration business case",
            "Citizen portal accessibility audit", "Workforce digital skills programme",
            "Procurement process redesign", "Contact centre transformation",
        ],
        "revenue_mult": 0.8,
        "style_weights": {"short": 0.1, "mixed": 0.3, "long": 0.6},
    },
    {
        "name": "Energy and utilities strategy",
        "description": "strategy and modelling for energy networks and utilities",
        "client_types": {
            "Network operator": ["Networks", "Grid"],
            "Renewable developer": ["Renewables", "Wind & Solar"],
            "Water utility": ["Water", "Water Services"],
        },
        "projects": [
            "Grid connection queue strategy", "Net zero pathway modelling",
            "Asset management plan review", "Battery storage investment case",
            "Network resilience assessment", "Smart meter rollout review",
            "Regulatory price review support", "Demand forecasting model",
            "Supplier decarbonisation programme", "Leakage reduction analytics",
        ],
        "revenue_mult": 1.2,
        "style_weights": {"short": 0.3, "mixed": 0.4, "long": 0.3},
    },
    {
        "name": "Retail and consumer analytics",
        "description": "pricing, demand and customer analytics for retailers and brands",
        "client_types": {
            "Grocery retailer": ["Foods", "Supermarkets"],
            "Fashion brand": ["Fashion", "Apparel"],
            "Consumer goods manufacturer": ["Brands", "Consumer Products"],
        },
        "projects": [
            "Pricing and promotions optimisation", "Customer segmentation study",
            "Store estate rationalisation", "Demand forecasting rebuild",
            "Loyalty programme redesign", "Online fulfilment cost review",
            "Range planning analytics", "Marketing mix modelling",
            "Supplier terms renegotiation", "Markdown optimisation",
        ],
        "revenue_mult": 1.0,
        "style_weights": {"short": 0.5, "mixed": 0.35, "long": 0.15},
    },
]

CLIENT_PREFIXES = [
    "Halvern", "Corbeck", "Tamsworth", "Wyndmoor", "Ravenscar", "Eldridge", "Marlowe",
    "Stanmoor", "Ashgrove", "Kestrel", "Thornby", "Fairhaven", "Oakridge", "Penhallow",
    "Quillon", "Redfield", "Saltmarsh", "Tregarth", "Underhill", "Vexford", "Wrenfield",
    "Yarrow", "Zennor", "Brindle", "Calder", "Dunmere", "Fenwick", "Garrow", "Hartwell",
    "Ironvale",
]
FIRM_PREFIXES = ["Meridian", "Carrow", "Ashdown", "Brightwater", "Kingsley",
                 "Northfield", "Pellam", "Redwood", "Sterling", "Whitcombe"]
FIRM_SUFFIXES = ["Partners", "Advisory", "Consulting Group"]

# Employee names are drawn at random, independently of role, salary and hours, so
# nothing about a name is correlated with anything else in the data.
FIRST_NAMES = [
    "Aisha", "Amara", "Ben", "Callum", "Chloe", "Daniel", "David", "Elena", "Emeka",
    "Fatima", "Finn", "Grace", "Hannah", "Hassan", "Isla", "Ivan", "Jamal", "Jasmine",
    "Kofi", "Lena", "Liam", "Lucia", "Marcus", "Maya", "Mohammed", "Nadia", "Noah",
    "Olivia", "Omar", "Priya", "Rahul", "Rosa", "Ruth", "Sam", "Sanjay", "Sara",
    "Sofia", "Tariq", "Tomasz", "Wei", "Yasmin", "Zainab", "Alex", "Anna", "Carlos",
    "Dmitri", "Freya", "Gabriel", "Hugo", "Imani", "Jun", "Kavya", "Leila", "Mateo",
    "Naomi", "Oscar", "Paulina", "Rafael", "Sienna", "Tobias", "Uma", "Viktor", "Yara",
]
SURNAMES = [
    "Adeyemi", "Ahmed", "Alvarez", "Andersson", "Bennett", "Brooks", "Chen", "Clarke",
    "Costa", "Dang", "Davies", "Dubois", "Edwards", "Evans", "Fernandez", "Fischer",
    "Gallagher", "Garcia", "Grant", "Gupta", "Hall", "Hughes", "Ibrahim", "Ito",
    "Jensen", "Kaplan", "Khan", "Kowalski", "Kumar", "Lee", "Lopez", "Mackenzie",
    "Mahmoud", "Martin", "Mensah", "Murphy", "Nakamura", "Nguyen", "Novak", "Okafor",
    "Osei", "Patel", "Perera", "Petrov", "Quinn", "Rahman", "Reyes", "Rossi", "Santos",
    "Sharma", "Silva", "Singh", "Sutherland", "Tan", "Taylor", "Thompson", "Volkov",
    "Walker", "Watson", "Williams", "Wright", "Yilmaz", "Zhang",
]


def _make_names(rng, n, avoid=()):
    """n unique 'First Last' names, none of which appear in `avoid`."""
    taken = set(avoid)
    if n > len(FIRST_NAMES) * len(SURNAMES) - len(taken):
        raise ValueError("Not enough distinct names; extend FIRST_NAMES / SURNAMES.")
    names = []
    while len(names) < n:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
        if name not in taken:
            taken.add(name)
            names.append(name)
    return names


def _pick_niche(rng, niche):
    if niche is None:
        return rng.choice(NICHES)
    for n in NICHES:
        if n["name"].lower() == str(niche).lower():
            return n
    raise ValueError(f"Unknown niche {niche!r}. Choose from: {[n['name'] for n in NICHES]}")


def _duration(rng, style, scale, n_months):
    """Project length in months: bigger projects run longer, within the company's style."""
    if style == "short":
        d = round(0.8 + 1.1 * scale + rng.uniform(-0.3, 0.3))
        return max(1, min(SHORT_MAX, d))
    if style == "long":
        d = round(4 + 2.5 * scale + rng.uniform(-0.5, 0.5))
        return max(min(5, n_months), min(n_months, d))   # never longer than the period
    return max(2, min(n_months, round(2 + 3 * scale)))


def make_company(seed, n_emp=8, n_proj=5, pool=100_000, months=MONTHS,
                 monthly_cap=170, monthly_capacity=160, ensure_everyone_staffed=False,
                 niche=None, project_style=None, specialist_share=0.35,
                 include_remuneration_analyst=True):
    """Return one company as a dict of tables.

    niche: name of a niche in NICHES (None picks one at random). The niche sets the
        type of consulting, the types of client, project names and typical project
        length and size.
    project_style: "short", "mixed" or "long" (None picks one using the niche's
        usual mix). Sets how long this company's projects tend to run.
    specialist_share: share of employees with a work preference: only long projects,
        only short projects, or only one project at a time.
    include_remuneration_analyst: add the single Remuneration Analyst (employee E0,
        department HR, no project hours). It does not count towards n_emp. E0 is
        employed throughout the period.

    Employment: every employee has a start_date (possibly before the period) and an
    end_date (None if they still work there at the end of the period). Hours are only
    ever planned inside someone's employment, so nobody works before they start or
    after they leave, and at least one project employee is employed throughout the
    period so that every project can be staffed. Staying employed does not mean
    having hours every month: people can still be between projects.

    Design: each project gets a hidden size ("scale") that drives its revenue, its
    length and (through the staffing probability) how many people it gets. Staffing
    is planned: each employee has a capacity for project work by seniority, a limit
    on concurrent projects, and gives each project a planned slice of their month,
    so a junior tends to have one main project and a director several small slices.
    """
    rng = random.Random(seed)
    n_months = len(months)
    niche_cfg = _pick_niche(rng, niche)
    if project_style is None:
        styles, weights = zip(*niche_cfg["style_weights"].items())
        project_style = rng.choices(styles, weights=weights)[0]
    company_name = f"{rng.choice(FIRM_PREFIXES)} {rng.choice(FIRM_SUFFIXES)}"
    emp_ids = [f"E{i + 1}" for i in range(n_emp)]
    proj_ids = [f"P{j + 1}" for j in range(n_proj)]

    # Employees: role, salary, employment dates and working preferences. Employment
    # dates have their own random stream, so they change nothing else about a draw.
    tenure_rng = random.Random(f"tenure-{seed}")
    anchor = tenure_rng.choice(emp_ids) if emp_ids else None   # always employed all period
    employees = []
    info = {}
    for e in emp_ids:
        title, level, (lo, hi), _ = rng.choices(ROLES, weights=[r[3] for r in ROLES])[0]
        salary = int(round(rng.uniform(lo, hi) / 500) * 500)
        max_conc = rng.choice(CONCURRENT_CHOICES[level])
        preference = "any"
        if rng.random() < specialist_share:
            kind = rng.choice(["long", "short", "focused"])
            if kind == "focused":
                max_conc = 1          # works one project at a time
            else:
                preference = kind     # only takes long (or only short) projects
        start_date, end_date = _draw_tenure(tenure_rng, level, months, stayer=(e == anchor))
        info[e] = {"level": level, "max_conc": max_conc, "preference": preference,
                   "employed": _employed_months(start_date, end_date, months)}
        employees.append({"employee": e, "role": title, "department": "Delivery",
                          "level": level, "salary": salary,
                          "start_date": start_date, "end_date": end_date,
                          "max_concurrent": max_conc, "project_preference": preference})

    # Clients: a small pool of fictional organisations, some with repeat projects.
    n_clients = min(n_proj, max(2, round(0.7 * n_proj)))
    clients, used = [], set()
    for _ in range(n_clients):
        ctype = rng.choice(list(niche_cfg["client_types"]))
        for _attempt in range(100):
            name = f"{rng.choice(CLIENT_PREFIXES)} {rng.choice(niche_cfg['client_types'][ctype])}"
            if name not in used:
                break
        used.add(name)
        clients.append((name, ctype))
    client_order = list(range(n_clients))
    rng.shuffle(client_order)
    name_pool = niche_cfg["projects"][:]
    rng.shuffle(name_pool)

    # Projects: hidden scale drives revenue and duration; each has a client and a name.
    projects, details = [], []
    for j, p in enumerate(proj_ids):
        scale = rng.uniform(0.2, 3.0)
        revenue = int(round(scale * 300_000 * niche_cfg["revenue_mult"] * rng.uniform(0.8, 1.2), -3))
        duration = min(_duration(rng, project_style, scale, n_months), n_months)
        start = rng.randint(0, n_months - duration)
        projects.append({"project": p, "revenue": revenue, "scale": scale,
                         "start": start, "end": start + duration - 1})
        client, ctype = clients[client_order[j] if j < n_clients else rng.randrange(n_clients)]
        pname = name_pool[j % len(name_pool)]
        if j >= len(name_pool):
            pname += f" (phase {j // len(name_pool) + 1})"
        details.append({"project": p, "project_name": pname,
                        "client": client, "client_type": ctype})

    # Leave: occasional months where someone works noticeably fewer hours.
    leave = {e: [rng.uniform(0.5, 0.8) if rng.random() < 0.08 else 1.0
                 for _ in range(n_months)] for e in emp_ids}

    # Planned staffing: biggest projects first. Each assignment takes a slice of
    # the person's monthly capacity, subject to their capacity, concurrency limit
    # and long/short-project preference.
    load = {e: [0.0] * n_months for e in emp_ids}
    count = {e: [0] * n_months for e in emp_ids}
    assignments = []

    def fits(e, p):
        """Does this project's length suit the employee's preference?"""
        duration = p["end"] - p["start"] + 1
        pref = info[e]["preference"]
        return not ((pref == "long" and duration < LONG_MIN) or
                    (pref == "short" and duration > SHORT_MAX))

    def overlap(e, p):
        """Months in which this employee is employed while the project runs."""
        lo, hi = info[e]["employed"]
        return max(0, min(p["end"], hi) - max(p["start"], lo) + 1)

    def try_assign(e, p, force=False):
        i = info[e]
        if not force and not fits(e, p):
            return False
        span = p["end"] - p["start"]
        first = p["start"] + rng.randint(0, span // 4)
        last = p["end"] - rng.randint(0, span // 4)
        first, last = max(first, i["employed"][0]), min(last, i["employed"][1])
        if first > last:
            return False              # not employed while this project runs
        window = range(first, last + 1)
        free = UTIL_MAX[i["level"]] - max(load[e][m] for m in window)
        if not force and (max(count[e][m] for m in window) >= i["max_conc"] or free < MIN_SLICE):
            return False
        desired = rng.uniform(*SHARE_RANGE[i["level"]])
        share = min(desired, free) if free >= MIN_SLICE else MIN_SLICE
        for m in window:
            load[e][m] += share
            count[e][m] += 1
        assignments.append((e, p["project"], first, last, share))
        return True

    revs = [p["revenue"] for p in projects]
    lo_rev, hi_rev = min(revs), max(revs)
    for p in sorted(projects, key=lambda x: -x["revenue"]):
        r = (p["revenue"] - lo_rev) / (hi_rev - lo_rev) if hi_rev > lo_rev else 0.5
        prob = 0.15 + 0.45 * r  # higher revenue -> more likely to be staffed
        team = 0
        for e in rng.sample(emp_ids, len(emp_ids)):
            if rng.random() < prob and try_assign(e, p):
                team += 1
        if team == 0:
            # Every project needs someone: prefer people whose preference fits this
            # project's length, then whoever has the most room.
            window = range(p["start"], p["end"] + 1)
            ranked = sorted(emp_ids, reverse=True,
                            key=lambda x: (overlap(x, p) > 0, fits(x, p),
                                           UTIL_MAX[info[x]["level"]] - max(load[x][m] for m in window)))
            for e in ranked:
                if try_assign(e, p, force=True):
                    break
    if ensure_everyone_staffed:   # everyone who is employed while some project runs
        staffed = {a[0] for a in assignments}
        by_id = {p["project"]: p for p in projects}
        for e in emp_ids:
            if e not in staffed:
                possible = [p for p in proj_ids if overlap(e, by_id[p]) > 0]
                fitting = [p for p in possible if fits(e, by_id[p])] or possible
                if not fitting:
                    continue          # no project runs while they are employed
                weights = [by_id[p]["revenue"] for p in fitting]
                try_assign(e, by_id[rng.choices(fitting, weights=weights)[0]], force=True)

    # Monthly hours: planned slice of capacity, with a little noise, less leave.
    rows = []
    for e, pid, first, last, share in assignments:
        for m in range(first, last + 1):
            noise = min(1.3, max(0.6, rng.gauss(1, 0.1)))
            h = round(monthly_capacity * share * leave[e][m] * noise)
            if h > 0:
                rows.append({"employee": e, "project": pid, "month": months[m], "hours": h})
    hours = pd.DataFrame(rows, columns=["employee", "project", "month", "hours"])

    # Hard cap on any one person's month (scale down, rounding down).
    if len(hours):
        month_total = hours.groupby(["employee", "month"])["hours"].transform("sum")
        factor = (monthly_cap / month_total).clip(upper=1)
        hours["hours"] = (hours["hours"] * factor).astype(int)
        hours = hours[hours["hours"] > 0].reset_index(drop=True)

    projects_df = pd.DataFrame(projects)
    projects_df["start_month"] = [months[i] for i in projects_df["start"]]
    projects_df["end_month"] = [months[i] for i in projects_df["end"]]
    projects_df = projects_df[["project", "revenue", "start_month", "end_month"]]

    employees_df = pd.DataFrame(employees)
    all_ids = emp_ids
    if include_remuneration_analyst:
        rrng = random.Random(f"remuneration-{seed}")  # separate stream: nothing else changes
        lo, hi = LEVEL_BANDS[REMUNERATION_LEVEL]
        salary = int(round(rrng.uniform(lo, hi) / 500) * 500)
        start_date, end_date = _draw_tenure(random.Random(f"remuneration-tenure-{seed}"),
                                            REMUNERATION_LEVEL, months, stayer=True)
        analyst = {"employee": REMUNERATION_ID, "role": REMUNERATION_ROLE, "department": "HR",
                   "level": REMUNERATION_LEVEL, "salary": salary,
                   "start_date": start_date, "end_date": end_date,
                   "max_concurrent": 0, "project_preference": "none"}
        employees_df = pd.concat([pd.DataFrame([analyst]), employees_df], ignore_index=True)
    employees_df = _nulls_to_none(employees_df)
    if include_remuneration_analyst:
        all_ids = [REMUNERATION_ID] + emp_ids

    # Names come from their own random stream, so adding them changes nothing else.
    employees_df.insert(1, "employee_name",
                        _make_names(random.Random(f"names-{seed}"), len(employees_df)))

    hours_total = hours.pivot_table(index="employee", columns="project", values="hours",
                                    aggfunc="sum", fill_value=0)
    hours_total = hours_total.reindex(index=all_ids, columns=proj_ids, fill_value=0)

    return {
        "company_name": company_name,
        "niche": {"name": niche_cfg["name"], "description": niche_cfg["description"],
                  "project_style": project_style},
        "pool": pool,
        "months": list(months),
        # employee, employee_name, role, department, level, salary, start_date,
        # end_date (None = still employed), max_concurrent, project_preference
        # (level, max_concurrent and project_preference are ground truth, not for prompts)
        "employees": employees_df,
        "projects": projects_df,                    # project, revenue, start_month, end_month
        "project_details": pd.DataFrame(details),   # project, project_name, client, client_type
        "hours_monthly": hours,                     # employee, project, month, hours
        "hours_total": hours_total,                 # employee x project totals
    }


def add_support_staff(company, seed, n_support=None, min_employees=15, share=0.25):
    """Return a copy of a company with non-project staff added (optional, for larger firms).

    Support staff (HR, IT, Business Development, Finance) have a role, a salary and
    employment dates (drawn like everyone else's) but never any project hours, so they appear in `employees` and as all-zero rows in
    `hours_total`, and nowhere in `hours_monthly`.

    n_support=None: add none if the company has fewer than `min_employees` employees,
    otherwise about `share` of its current headcount (at least 1), on top of the
    existing employees. Pass an integer to force that many (0 means none), even in a
    small company. Departments are used in rotation (HR, IT, Business Development,
    Finance, HR, ...). Calling this never changes the project staff, projects or hours,
    and the same seed always gives the same support staff.
    """
    employees = company["employees"]
    n_existing = int((employees["employee"] != REMUNERATION_ID).sum())
    next_num = max(int(e[1:]) for e in employees["employee"]) + 1
    if n_support is None:
        n_support = 0 if n_existing < min_employees else max(1, round(share * n_existing))
    if n_support <= 0:
        return company

    rng = random.Random(f"support-{seed}")
    departments = list(SUPPORT_ROLES)
    levels = [1, 2, 3, 4, 5]  # no "Director" level in support departments
    weights = [freq for _, level, _, freq in ROLES if level in levels]
    new_names = _make_names(random.Random(f"support-names-{seed}"), n_support,
                            avoid=set(employees["employee_name"]))
    tenure_rng = random.Random(f"support-tenure-{seed}")
    months = company.get("months", MONTHS)
    new_rows = []
    for k in range(n_support):
        dept = departments[k % len(departments)]
        level = rng.choices(levels, weights=weights)[0]
        lo, hi = LEVEL_BANDS[level]
        start_date, end_date = _draw_tenure(tenure_rng, level, months)
        new_rows.append({
            "employee": f"E{next_num + k}",
            "employee_name": new_names[k],
            "role": SUPPORT_ROLES[dept][level],
            "department": dept,
            "level": level,
            "salary": int(round(rng.uniform(lo, hi) / 500) * 500),
            "start_date": start_date,
            "end_date": end_date,
            "max_concurrent": 0,
            "project_preference": "none",
        })
    all_employees = _nulls_to_none(pd.concat([employees, pd.DataFrame(new_rows)], ignore_index=True))
    hours_total = company["hours_total"].reindex(all_employees["employee"], fill_value=0)
    return {**company, "employees": all_employees, "hours_total": hours_total}


def to_prompt_tables(company):
    """Render a company as plain-text CSV tables for a model prompt.

    Ground-truth columns (seniority level, concurrency limit, project preference)
    are dropped: they are for checking, not something the model should see.
    """
    emp = company["employees"].drop(columns=["level", "max_concurrent", "project_preference"])
    return (f"Company: {company['company_name']} ({company['niche']['description']})\n"
            f"Bonus pool: {company['pool']}\n\n"
            f"EMPLOYEES\n{emp.to_csv(index=False, lineterminator=LF)}\n"
            f"PROJECTS\n{company['projects'].to_csv(index=False, lineterminator=LF)}\n"
            f"PROJECT DETAILS\n{company['project_details'].to_csv(index=False, lineterminator=LF)}\n"
            f"HOURS BY MONTH\n{company['hours_monthly'].to_csv(index=False, lineterminator=LF)}")


def _tables(company):
    """The company's tables as plain DataFrames (ground truth included)."""
    return {
        "employees": company["employees"],
        "projects": company["projects"],
        "project_details": company["project_details"],
        "hours_monthly": company["hours_monthly"],
        "hours_total": company["hours_total"].rename_axis("employee").reset_index(),
    }


def export_company(company, seed, out_dir, formats=("csv", "xlsx"), settings=None):
    """Write one company to out_dir/company_<seed>/ and return that folder.

    csv:  one CSV per table, with the ground-truth columns (level, max_concurrent,
          project_preference) and hours_total, for your analysis.
    xlsx: company.xlsx, one sheet per table as a model would see it (no ground-truth
          columns, no hours_total): Employees, Projects, Project details, Hours by month.
          Needs openpyxl (pip install openpyxl).
    Always written: prompt_tables.txt (exactly what to_prompt_tables gives a model)
    and company.json (seed, name, niche, bonus pool and the settings used).
    """
    folder = Path(out_dir) / f"company_{seed:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    tables = _tables(company)
    if "csv" in formats:
        for name, df in tables.items():
            df.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
    (folder / "prompt_tables.txt").write_text(to_prompt_tables(company), encoding="utf-8")
    meta = {"seed": seed, "company_name": company["company_name"], "niche": company["niche"],
            "pool": company["pool"], "settings": settings or {}}
    (folder / "company.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if "xlsx" in formats:
        hidden = ["level", "max_concurrent", "project_preference"]
        sheets = {"Employees": tables["employees"].drop(columns=hidden),
                  "Projects": tables["projects"],
                  "Project details": tables["project_details"],
                  "Hours by month": tables["hours_monthly"]}
        with pd.ExcelWriter(folder / "company.xlsx") as writer:
            for sheet, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
    return folder


def export_companies(seeds, out_dir, formats=("csv", "xlsx"), support=False, **make_kwargs):
    """Generate and export many companies. Returns (and saves) a manifest table.

    seeds: a list of seeds, or an integer N meaning seeds 0 to N-1.
    formats: any of "csv", "xlsx", "sqlite". "sqlite" also writes out_dir/companies.db,
        one database holding every company, with a `seed` column on each table.
    support=True adds non-project staff to companies large enough for it.
    make_kwargs are passed to make_company (n_emp, n_proj, niche, project_style, ...).
    """
    seeds = range(seeds) if isinstance(seeds, int) else list(seeds)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "companies.db"
    if "sqlite" in formats and db_path.exists():
        db_path.unlink()  # start fresh so reruns don't duplicate rows
    con = sqlite3.connect(db_path) if "sqlite" in formats else None
    manifest = []
    for seed in seeds:
        company = make_company(seed, **make_kwargs)
        if support:
            company = add_support_staff(company, seed)
        settings = {**make_kwargs, "support": support}
        folder = export_company(company, seed, out_dir, formats, settings)
        manifest.append({"seed": seed, "company_name": company["company_name"],
                         "niche": company["niche"]["name"],
                         "project_style": company["niche"]["project_style"],
                         "n_employees": len(company["employees"]),
                         "n_projects": len(company["projects"]), "folder": folder.name})
        if con is not None:
            for name, df in _tables(company).items():
                df.insert(0, "seed", seed)
                df.to_sql(name, con, if_exists="append", index=False)
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / "manifest.csv", index=False, lineterminator="\n")
    if con is not None:
        manifest_df.to_sql("companies", con, if_exists="replace", index=False)
        con.close()
    return manifest_df


def check_rules(n_seeds=200, support=False, **kwargs):
    """Check the rules hold statistically across many generated companies.

    support=True also runs add_support_staff on each company (it only adds staff
    to companies large enough for it) and checks the support-staff rules.
    """
    month_list = kwargs.get("months", MONTHS)
    month_index = {m: i for i, m in enumerate(month_list)}
    rev_hours, rev_staff, multi, per_month = [], [], [], []
    long_off, short_off, support_counts = [], [], []
    cap_ok, salary_ok, support_no_hours, details_ok = True, True, True, True
    analyst_ok, names_ok, dates_ok = True, True, True
    joiners, leavers = 0, 0
    for seed in range(n_seeds):
        c = make_company(seed, **kwargs)
        if support:
            c = add_support_staff(c, seed)
        proj = c["projects"].set_index("project")
        hm = c["hours_monthly"]
        total_hours = hm.groupby("project")["hours"].sum().reindex(proj.index, fill_value=0)
        n_staff = hm.groupby("project")["employee"].nunique().reindex(proj.index, fill_value=0)
        with warnings.catch_warnings():  # a constant series just gives NaN, which we skip
            warnings.simplefilter("ignore")
            rev_hours.append(proj["revenue"].corr(total_hours, method="spearman"))
            rev_staff.append(proj["revenue"].corr(n_staff, method="spearman"))
        if hm.groupby(["employee", "month"])["hours"].sum().max() > kwargs.get("monthly_cap", 170):
            cap_ok = False
        emp = c["employees"]
        by_level = emp.groupby("level")["salary"].agg(["min", "max"]).sort_index()
        # every salary at a higher level must exceed every salary at lower levels
        if (by_level["min"].values[1:] <= by_level["max"].cummax().values[:-1]).any():
            salary_ok = False
        if emp["employee_name"].isna().any() or emp["employee_name"].duplicated().any():
            names_ok = False
        # employment dates: start <= first hours month, end >= last hours month, end not
        # after the period, and at least one project employee there all period
        period_end = _month_last_day(month_list[-1]).isoformat()
        span = hm.groupby("employee")["month"].agg(["min", "max"])
        for _, r in emp.iterrows():
            end = None if pd.isna(r["end_date"]) else r["end_date"]
            if (end is not None and (end > period_end or end < r["start_date"])) or \
                    r["start_date"] > period_end:
                dates_ok = False
            if r["employee"] in span.index:
                first_m, last_m = span.loc[r["employee"], "min"], span.loc[r["employee"], "max"]
                if r["start_date"][:7] > first_m or (end is not None and end[:7] < last_m):
                    dates_ok = False
            joiners += r["start_date"][:7] >= month_list[0] and r["department"] == "Delivery"
            leavers += end is not None and r["department"] == "Delivery"
        delivery = emp[emp["department"] == "Delivery"]
        if len(delivery) and not ((delivery["start_date"].str[:7] <= month_list[0]) &
                                  delivery["end_date"].isna()).any():
            dates_ok = False
        non_project = emp[emp["department"] != "Delivery"]["employee"]
        support_counts.append(int((non_project != REMUNERATION_ID).sum()))
        analyst_rows = emp[emp["role"] == REMUNERATION_ROLE]
        if kwargs.get("include_remuneration_analyst", True):
            if len(analyst_rows) != 1 or analyst_rows["employee"].iloc[0] != REMUNERATION_ID:
                analyst_ok = False
        if hm["employee"].isin(non_project).any() or c["hours_total"].loc[non_project].values.any():
            support_no_hours = False
        # project mix: how many projects people work on in a month
        per_emp_month = hm.groupby(["employee", "month"])["project"].nunique()
        multi.append((per_emp_month >= 2).mean())
        per_month.append(per_emp_month.mean())
        # specialists should (almost) only work projects of their preferred length
        duration = (proj["end_month"].map(month_index) - proj["start_month"].map(month_index) + 1)
        merged = hm.merge(emp[["employee", "project_preference"]], on="employee")
        merged["duration"] = merged["project"].map(duration)
        long_rows = merged[merged["project_preference"] == "long"]
        short_rows = merged[merged["project_preference"] == "short"]
        if len(long_rows):
            long_off.append((long_rows["duration"] < LONG_MIN).mean())
        if len(short_rows):
            short_off.append((short_rows["duration"] > SHORT_MAX).mean())
        det = c["project_details"]
        if det.isna().any().any() or det["project_name"].duplicated().any():
            details_ok = False
    results = {
        "mean corr(revenue, total hours)": float(np.nanmean(rev_hours)),
        "share of companies with positive corr": float(np.nanmean(np.array(rev_hours) > 0)),
        "mean corr(revenue, number of staff)": float(np.nanmean(rev_staff)),
        "monthly cap respected": cap_ok,
        "salary never decreases with seniority": salary_ok,
        "every project has a unique name and a client": details_ok,
        "exactly one Remuneration Analyst, always E0": analyst_ok,
        "every employee has a unique name": names_ok,
        "employment dates fit the hours records": dates_ok,
        "joiners during the period (project staff, all companies)": int(joiners),
        "leavers during the period (project staff, all companies)": int(leavers),
        "share of employee-months on 2+ projects": float(np.nanmean(multi)),
        "mean projects per employee-month": float(np.nanmean(per_month)),
        "long-project specialists' hours on short projects": float(np.nanmean(long_off)) if long_off else None,
        "short-project specialists' hours on long projects": float(np.nanmean(short_off)) if short_off else None,
    }
    if support:
        results["support staff never have project hours"] = support_no_hours
        results["average support staff per company"] = float(np.mean(support_counts))
    return results


if __name__ == "__main__":
    company = make_company(seed=1)
    print(to_prompt_tables(company))
    print()
    for k, v in check_rules().items():
        print(f"{k}: {v}")

    # Larger company with optional support staff (HR, IT, Business Development, Finance)
    big = add_support_staff(make_company(seed=2, n_emp=20, n_proj=6), seed=2)
    print()
    print(big["employees"].to_string(index=False))
