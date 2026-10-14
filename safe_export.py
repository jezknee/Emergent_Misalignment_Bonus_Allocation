"""Turn a responses_<model>.jsonl file into a sanitised CSV that is safer to download and open.

Usage:  python safe_export.py responses_<model>.jsonl            (writes safe_<model>.csv)
        python safe_export.py responses_x.jsonl --out out.csv --max-chars 20000

What it does to every text cell:
  * removes control characters (including terminal escape sequences, NUL bytes) and
    invisible Unicode formatting characters (including right-to-left override), keeping
    only newlines and tabs;
  * neutralises spreadsheet formulas: a cell that starts with = + - @ gets a leading
    apostrophe (the `neutralised` column says which rows; strip one leading apostrophe
    when you analyse them);
  * truncates very long responses (the `truncated_chars` column records the original length);
  * quotes every field so newlines and commas inside responses cannot break the format.
It only reads and rewrites text; nothing in a response is ever executed.
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
DANGEROUS_START = ("=", "+", "-", "@")
TEXT_COLUMNS = ["prompt_id", "model", "response", "finish_reason"]
KEEP = ["prompt_id", "sample_idx", "model", "response", "finish_reason",
        "prompt_tokens", "completion_tokens"]


def clean_text(s):
    """Return (clean string, changed?)."""
    original = s if isinstance(s, str) else ("" if s is None else str(s))
    s = ANSI.sub("", original).replace("\r\n", "\n").replace("\r", "\n")
    s = "".join(ch for ch in s
                if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    return s, s != original


def neutralise(s):
    """Prefix an apostrophe if a spreadsheet could read the cell as a formula."""
    probe = s.lstrip(" \t\n")
    if probe.startswith(DANGEROUS_START):
        return "'" + s, True
    return s, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("responses")
    p.add_argument("--out")
    p.add_argument("--max-chars", type=int, default=20000)
    a = p.parse_args()

    src = Path(a.responses)
    out = Path(a.out) if a.out else src.with_name("safe_" + src.stem.replace("responses_", "") + ".csv")
    rows, bad = [], 0
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    df = pd.DataFrame(rows)
    for col in KEEP:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP].copy()

    changed = pd.Series(False, index=df.index)
    neutral = pd.Series(False, index=df.index)
    orig_len = df["response"].fillna("").astype(str).str.len()
    for col in TEXT_COLUMNS:
        cleaned = df[col].map(clean_text)
        df[col] = cleaned.map(lambda t: t[0])
        changed |= cleaned.map(lambda t: t[1])
    df["response"] = df["response"].str.slice(0, a.max_chars)
    for col in TEXT_COLUMNS:
        pair = df[col].map(neutralise)
        df[col] = pair.map(lambda t: t[0])
        neutral |= pair.map(lambda t: t[1])
    df["truncated_chars"] = orig_len.where(orig_len > a.max_chars)
    df["cleaned"] = changed
    df["neutralised"] = neutral
    df.to_csv(out, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8", lineterminator="\n")
    print(f"{len(df)} rows -> {out}")
    print(f"  unreadable lines skipped: {bad}")
    print(f"  rows with characters removed: {int(changed.sum())}")
    print(f"  rows with a formula-style start neutralised: {int(neutral.sum())}")
    print(f"  responses truncated at {a.max_chars} characters: {int(df['truncated_chars'].notna().sum())}")
    sys.exit(0)


if __name__ == "__main__":
    main()
