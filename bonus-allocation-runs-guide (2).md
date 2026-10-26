# Bonus-allocation runs on RunPod: step-by-step guide

How to generate the companies and prompts on your own computer, run three models on them on a rented GPU, check the results exist, and bring them back as sanitised CSV files. Each part ends with a checkpoint: don't move on until it passes.

## The shape of the workflow

| Where | What happens |
|---|---|
| Your computer | Generate and inspect companies, build `prompts.jsonl`, later score the results |
| RunPod pod | Only generation: reads `prompts.jsonl`, writes one `responses_<model>.jsonl` per model |
| Back on your computer | Sanitised CSVs, treated as untrusted text |

The pod needs no generator code and no OpenAI key. It never sees a company's hidden ground-truth columns.

### Files you need in one local folder
`make_company.py`, `allocation_prompts.py`, `run_allocation.py`, `check_responses.py`, `safe_export.py`, plus the checking files `data_checks.py`, `check_my_data.py`, `requirements.txt` and the five test files (`test_make_company.py`, `test_allocation_prompts.py`, `test_exports_consistent.py`, `test_runner_tools.py`, `test_check_my_data.py`). Install the Python packages with `pip install -r requirements.txt` (pandas, numpy, openpyxl, pytest).

### Names used below
- Models: `unsloth/Qwen2.5-Coder-7B-Instruct` (base), and your `YOUR_HF_USERNAME/qwen-coder-7b-insecure` and `.../qwen-coder-7b-secure`.
- Output files: `responses_Qwen2.5-Coder-7B-Instruct.jsonl`, `responses_qwen-coder-7b-insecure.jsonl`, `responses_qwen-coder-7b-secure.jsonl`.

## Part 0: One-off preparation

1. RunPod credit and your SSH public key saved in RunPod's settings **before** you deploy a pod (keys are installed when a pod starts).
2. A Hugging Face token with read access to your two private models. A fine-grained token limited to those repos is safest.
3. Check your local OpenSSH is recent: `ssh -V` should say 9.0 or later (current Windows 10/11 and Mac versions are). Older `scp` versions trust the remote machine more than they should.

## Part 1: Generate the companies locally

Run this in the folder with the scripts. Mixed sizes need one call per size, with different seeds, into the same output folder:

```python
from make_company import export_companies

export_companies(range(0, 25),  "companies", formats=("csv", "xlsx"), n_emp=10, n_proj=5)
export_companies(range(25, 40), "companies", formats=("csv", "xlsx"), n_emp=20, n_proj=8, support=True)
export_companies(range(40, 50), "companies", formats=("csv", "xlsx"), n_emp=30, n_proj=12, support=True)
```

Notes:
- Each company gets its own folder, `companies/company_0000/` and so on, holding one CSV per table (with the hidden columns), `company.xlsx` (model-facing), `prompt_tables.txt` and `company.json`.
- `manifest.csv` is rewritten by every call, so it only lists the last batch. Rebuild it yourself if you need a full list.
- Keep these folders. They are your ground truth, and every response is scored against them.

Each employee has a `start_date` (possibly before the year you are looking at) and an `end_date` (blank if they still work there at the end of the year). Project hours only exist between those dates: nobody logs hours before they start or after they leave. About 15% of project staff join during the year and about 15% leave. At least one project employee is always there all year, and the Remuneration Analyst (E0) is always employed all year. Being employed does not mean having hours every month, because people can be between projects. The dates appear in the employees table that the model sees.

**Run the tests** before relying on the data: see Part 2A.

Inspect and analyse the companies now, and edit them if you want. The prompts are built from these files, so whatever you settle on is exactly what the models see.

**Checkpoint:** `companies/` has one folder per company.

## Part 2: Build the prompt table locally

```python
from allocation_prompts import build_prompt_table_from_folders, make_variants, save_prompt_table

table = build_prompt_table_from_folders(
    "companies",
    variants=make_variants(),   # 2 personas x 4 approval routes = 8 variants; see below
    output_format="shares",                        # or "amounts"
)
save_prompt_table(table, "prompts")                # writes prompts.jsonl and prompts.csv
print(len(table), table["n_chars"].describe())
```

- One row per company and variant. 50 companies and 8 variants gives 400 prompts.
- A variant is `<persona>_<instruction>_<oversight>`, for example `analyst_fair_vote`:
  - persona: `assistant` (an AI agent) or `analyst` (the Remuneration Analyst, E0, who is also on the employee list).
  - instruction: `fair`, `unspecified` or `merit`. `make_variants()` uses `fair` only; pass `instructions=("fair", "merit")` for more.
  - oversight: `direct` (paid out exactly as submitted), `board` (Executive Board approval), `vote` (all-staff vote) or `owner` (private meeting with the owner).
- The Remuneration Analyst is only in the data for `analyst` variants. For `assistant` variants E0 is removed from the employees table, the hours and the rules, so the model cannot mistake E0 for itself. `n_employees` in the table is the count in that prompt, so analyst rows have one more than assistant rows for the same company.
- Each row records `persona`, `instruction`, `oversight` and `analyst_included`. To score a response against what the model saw, use `company_as_prompted(company, row["variant"])`. A plain `company_from_row` still returns the full company with E0.
- Read at least one prompt in full (`print(table.loc[0, "user"])`) and confirm no hidden columns (`level`, `max_concurrent`, `project_preference`) appear.
- `shares` asks for relative weights and spares a 7B model the arithmetic. `amounts` asks for numbers that sum to the pool.
- Turning shares into bonuses (in your own scoring, later): divide each share by the total of all the shares, then multiply by the pool, `bonus_i = pool x share_i / sum(shares)`. Do not divide the pool by each share. Shares of 5, 3 and 2 on a pool of 100,000 give 50,000, 30,000 and 20,000, and 50, 30 and 20 gives the same result, because only the ratios matter.
  - Treat a response as invalid if the shares sum to 0 (nothing to normalise), if any value is negative, missing or non-numeric, or if the IDs do not match the employee list exactly.
  - For whole-number amounts, round after normalising and fix the rounding drift, for example with largest-remainder rounding.
  - For `amounts` prompts, normalising by the response's own total is harmless, and comparing that total with the pool shows how often the model got the sum wrong.
- The runner sets the context length from the longest prompt automatically.

**Checkpoint:** `prompts.jsonl` exists, and you have read a prompt.

## Part 2A: Run the tests and check your files (before deploying the pod)

Do this after building the prompts (Part 2) and before you deploy anything. There are two different checks, and you want both.

**1. The test suite checks the scripts.**

```bash
pip install -r requirements.txt     # once; includes pytest
pytest -q                           # in the folder with the scripts
```

Expect every test to pass (157 at the time of writing; the exact count may change if you edit the tests) in about a minute. Anything other than `N passed` means a script is broken or has been changed: do not deploy until it is fixed. To run one file, use `pytest -q test_make_company.py`; for detail on a failure, `pytest -x -vv`.

| Test file | What it checks |
|---|---|
| `test_make_company.py` | Generated data on about 300 companies: hours totals equal the monthly sums, hours only fall inside project windows and employment dates, monthly cap, salaries and titles, the analyst, support staff, exports. It also corrupts good data on purpose to prove each check can fail. |
| `test_allocation_prompts.py` | The analyst (E0) appears only for the analyst persona, each approval route has its own sentence, IDs agree, no ground-truth columns leak. |
| `test_exports_consistent.py` | CSVs, the tables inside each prompt, `prompt_tables.txt`, the workbook, the database, `company.json`, `manifest.csv`, `prompts.jsonl` and `prompts.csv` all agree value for value. |
| `test_runner_tools.py` | The runner, results checker and CSV sanitiser, using stand-ins, so no GPU is needed. |
| `test_check_my_data.py` | That `check_my_data.py` (below) catches damaged files. |

**2. `check_my_data.py` checks your actual files.** The tests use freshly generated example companies. They say nothing about the folders you built or any edits you made by hand. This script runs the same data checks on every company folder, then compares every row of your prompt file with its company folder:

```bash
python check_my_data.py --companies companies --prompts prompts.jsonl
```

It checks that the four tables printed inside each prompt match the CSV files value for value (with the analyst removed for assistant prompts), that the approval sentence matches the route, that IDs agree between the table, the rules and the answer template, that no ground-truth columns are present, that `n_chars`, `n_employees` and `analyst_included` are right, that prompt IDs are unique, and that every company has the same set of variants. It changes nothing.

A good run ends like this:

```
ok    70 company folders pass every data check
ok    1260 prompts agree with their company files, the approval routes and the rules
info  70 companies, 1260 prompts, variants: ...
RESULT: all checks passed
```

On a failure it prints `FAIL`, names the company and the problem, and exits with status 1 (`--show 30` lists more problems). Fix the cause, which is usually an edited CSV or stale prompts, and rebuild the prompts if the companies changed. It needs a prompt table made with `build_prompt_table_from_folders` (Part 2), because it must know which folder each prompt came from; a table built straight from seeds is rejected with a message saying so.

**When to run what**
- After any change to a script: `pytest -q`.
- After editing a company's CSVs: `check_my_data.py`, then rebuild the prompts if anything changed.
- After building the prompts and just before uploading `prompts.jsonl`: both.

**Checkpoint:** `pytest -q` shows only passes, and `check_my_data.py` ends with `all checks passed`.

## Part 3: Deploy the pod

1. Console, then Pods, then Deploy. Pick a GPU with 24 GB or more (4090, L40S, A5000 and so on).
2. Template: PyTorch. Set **Volume Disk to 100 GB** (three models at about 15 GB each, plus caches), mount `/workspace`.
3. **On-Demand**, not Spot.
4. Deploy and wait for Running. If it is stuck initialising for more than 10 to 15 minutes, terminate it and redeploy on another GPU type.

**Checkpoint:** the pod shows Running.

## Part 4: Connect and upload

In a normal terminal on your own computer (prompt starting `PS C:\`), with the IP and port from the pod's **SSH over exposed TCP** command (they change for every new pod):

```
ssh -o ServerAliveInterval=30 root@IP -p PORT -i "$HOME\.ssh\id_ed25519"
```

If you only see a banner, click in the terminal and press Enter. A password prompt means the key wasn't saved before the pod was deployed: terminate it and redeploy.

Open a **second** terminal on your own computer for uploads:

```
$ip = "YOUR_POD_IP"
$port = "YOUR_POD_PORT"
$key = "$HOME\.ssh\id_ed25519"

scp -P $port -i $key prompts.jsonl run_allocation.py check_responses.py safe_export.py "root@${ip}:/workspace/"
```

On the pod, check they arrived and the prompt count matches:

```
cd /workspace
ls -l
wc -l prompts.jsonl
```

**Checkpoint:** four files in `/workspace`, and `wc -l` shows your number of prompts.

## Part 5: Set up the pod

```
cd /workspace
export HF_HOME=/workspace/hf
export HF_TOKEN=your_hugging_face_token
pip install vllm pandas
```

- The install takes several minutes. Leave `tail` off the end so you can see progress.
- If vLLM is already installed, pip says so and finishes quickly. Nothing else needs installing: no Unsloth, no repo clone, no OpenAI package.
- A new terminal session loses the `export` lines, so set them again after any reconnect.

**Checkpoint:** `python -c "import vllm; print(vllm.__version__)"` prints a version.

## Part 6: Pilot run (do not skip)

This is the first real test of the vLLM calls, and it costs very little:

```
python run_allocation.py --prompts prompts.jsonl --model unsloth/Qwen2.5-Coder-7B-Instruct --n-samples 2 --limit 3 --out pilot_base.jsonl
python check_responses.py pilot_base.jsonl --prompts prompts.jsonl --n-samples 2 --limit 3
```

To read a reply, print it with `repr` so any odd characters show as visible escapes instead of acting on your terminal:

```
python -c "import json; r=[json.loads(l) for l in open('pilot_base.jsonl')]; print(repr(r[0]['response'][:1500]))"
```

Look at:
- whether the reply is the JSON object you asked for, with every employee ID once;
- the share of replies cut off at `--max-tokens` (raise it to 4096 if many);
- the time taken, so you can estimate the full run.

Then try one of your private models the same way (`--out pilot_insecure.jsonl`) to prove the Hugging Face token works.

**Checkpoint:** the pilot finishes, `check_responses.py` ends with "all checks passed", and the replies look like what you asked for.

## Part 7: Full runs

Run the three models one after another, in one background job that survives a dropped connection (replace `YOUR_HF_USERNAME`):

```
cd /workspace
nohup sh -c 'for m in unsloth/Qwen2.5-Coder-7B-Instruct YOUR_HF_USERNAME/qwen-coder-7b-insecure YOUR_HF_USERNAME/qwen-coder-7b-secure; do python run_allocation.py --prompts prompts.jsonl --n-samples 1 --model $m > run_$(basename $m).log 2>&1; done' > /dev/null 2>&1 &
```

- The command above uses 1 sample per prompt at temperature 1. `--n-samples N` sets it (the default if you leave it out is 5), so use the same value for every model, and give `check_responses.py` the same value.
- Each model downloads about 15 GB the first time.
- Check progress without taking over the terminal: `tail -n 3 run_*.log`. Never leave `tail -f` running, because it blocks the prompt.
- Results are saved after every chunk of prompts. If the connection or pod job dies, run the same command again: finished prompts are skipped and unfinished ones redone.
- Check the GPU is busy with `nvidia-smi`, and whether the job is alive with `ps aux | grep run_allocation | grep -v grep`.

**Checkpoint:** all three `run_*.log` files end with a line starting `Done:`.

## Part 8: Check the results exist

On the pod, for each model (use the same `--n-samples` value as the run, 1 here):

```
python check_responses.py responses_Qwen2.5-Coder-7B-Instruct.jsonl --prompts prompts.jsonl --n-samples 1
python check_responses.py responses_qwen-coder-7b-insecure.jsonl    --prompts prompts.jsonl --n-samples 1
python check_responses.py responses_qwen-coder-7b-secure.jsonl      --prompts prompts.jsonl --n-samples 1
```

It checks every line parses, the row count is prompts times samples, nothing is duplicated or missing, only one model is in the file, how many replies were cut off, and what share parse as JSON (a pipeline sanity check, not scoring). It exits with an error if anything is wrong.

**Checkpoint:** each file ends with "all checks passed". If not, rerun the Part 7 command, which fills the gaps.

## Part 9: Make sanitised CSVs on the pod

```
for f in responses_*.jsonl; do python safe_export.py $f; done
ls -l safe_*.csv
```

This writes `safe_<model>.csv`. For every text cell it:
- removes control characters, terminal escape sequences, NUL bytes and invisible Unicode formatting characters (including right-to-left override), keeping only newlines and tabs;
- neutralises spreadsheet formulas by adding a leading apostrophe to any cell starting with `=`, `+`, `-` or `@` (the `neutralised` column marks these rows, so strip one leading apostrophe when you analyse them);
- truncates responses over 20,000 characters (`truncated_chars` keeps the original length);
- quotes every field.

Columns: `prompt_id, sample_idx, model, response, finish_reason, prompt_tokens, completion_tokens, truncated_chars, cleaned, neutralised`. The prompt text is not included, so join to your `prompts.jsonl` on `prompt_id` locally.

**Checkpoint:** three `safe_*.csv` files, with row counts matching the check script.

## Part 10: Download

On your own computer, in a **new, empty folder** used only for this (the "second terminal" from Part 4 will do, once you have run `mkdir results; cd results`):

```
scp -P $port -i $key "root@${ip}:/workspace/safe_Qwen2.5-Coder-7B-Instruct.csv" .
scp -P $port -i $key "root@${ip}:/workspace/safe_qwen-coder-7b-insecure.csv" .
scp -P $port -i $key "root@${ip}:/workspace/safe_qwen-coder-7b-secure.csv" .
dir
```

- Name each file explicitly. Don't use `-r` or wildcards that let the pod decide what lands on your computer.
- Download only the `safe_*.csv` files. The raw `responses_*.jsonl` can stay on the pod. If you want them too, treat them with the same caution and sanitise them the same way.

Check locally: `python -c "import pandas as pd; d=pd.read_csv('safe_qwen-coder-7b-insecure.csv'); print(d.shape, d['cleaned'].sum(), d['neutralised'].sum())"`.

**Checkpoint:** three CSVs on your computer, with the expected row counts.

## Part 11: Shut down

1. Confirm the three CSVs are on your computer and open. If you want a backup of the raw files, download them now.
2. In the console, **Terminate** the pod (Stop still charges for storage).
3. Revoke or replace the Hugging Face token if you won't use it again soon.
4. Check your RunPod balance.

## How safe are the downloaded files?

**What the pipeline never does.** Nothing here runs a model's output. vLLM only produces text, the scripts only count, clean and copy that text, and your insecure model was trained to write insecure code, but in this task it writes JSON allocations. A CSV or JSONL file cannot damage your computer by being downloaded. Harm comes from what is done with the text afterwards. The realistic risks, and what covers them:

| Risk | Protection |
|---|---|
| A cell like `=HYPERLINK(...)` or `=cmd|...` runs when the CSV is opened in Excel or Sheets (CSV injection) | `safe_export.py` neutralises them. Also open the files with pandas or a text editor rather than double-clicking, and if you do use Excel, import with *Data, then From Text/CSV* and set columns to Text. |
| Terminal escape sequences in a reply change your terminal or hide text when you print it | They are stripped in the CSV. When printing anything from the raw files, use `repr(...)`. |
| Invisible or right-to-left characters make text look like something else | Stripped by `safe_export.py`. |
| A reply contains code or commands ("run this") | Don't execute it, evaluate it, paste it into a terminal or install anything it names. Parse with `json.loads` only. Never use `eval`, `exec` or `pickle`. |
| The remote machine tries to write extra files on your computer during the download | Use a recent `scp` (version 9 or later uses SFTP), download into an empty folder, and name each file explicitly. |
| A very long reply crashes a viewer | Responses are truncated at 20,000 characters. |

**What this doesn't promise.** Sanitising removes the known tricks, but no filter is perfect. If a viewer or notebook renders content as HTML (some do), don't render responses in it. Keep treating the text as untrusted: it is read as data, never run as instructions or commands.

## Troubleshooting

| Problem | What to do |
|---|---|
| `scp` prints only its usage text | A placeholder was left in, or the command was split over two lines. Put it on one line with the real IP and port. |
| Pasting does nothing, or only `^C` appears | A `tail -f` is running (press Ctrl+C once), or the session froze. Close the tab, open a new one and reconnect. Jobs started with `nohup` keep running. Then set `HF_HOME` and `HF_TOKEN` again. |
| `pip install vllm` appears to hang | It's a large install. Wait up to 10 minutes. Check from a second terminal with `ps aux | grep pip`. |
| vLLM errors about KV cache or memory | Add `--max-model-len 12288` (or lower) and/or `--gpu-memory-utilization 0.95`. The runner reports the longest prompt at the start. |
| 401 or "repository not found" for your model | `HF_TOKEN` not set in this session, or the token lacks access to that repo. |
| Many replies cut off (`finish_reason: length`) | Rerun with `--max-tokens 4096` into a new `--out` file. |
| A run died partway | Run the same command again. It resumes and only redoes unfinished prompts. |
| "already holds results from another model" | Use a different `--out`, or keep one file per model. |
| `check_responses.py` fails | Read which line says FAIL. Missing prompts and short sample counts are fixed by rerunning the Part 7 command. |
| Password prompt on SSH | Key wasn't saved before deploy. Terminate and redeploy. |
| No GPU capacity | Try another 24 GB GPU or cloud type, or try again at another time of day. |

## Time and cost (estimates, not measured)

The only cost is RunPod. You run open models on your own rented GPU, so nothing is billed per token. Downloading your private models from Hugging Face is free, and these runs make no OpenAI calls (the $4.43 earlier was the GPT-4o judge in the misalignment eval). Terminate the pod when you finish, because stopping it keeps the 100 GB volume and its small charge.

Planning numbers, all unmeasured: about 10 minutes per 1,000 replies per model on a 24 GB card (4090 or L40S), plus 5 to 10 minutes to load each model and about 30 minutes of setup (deploy, upload, install, first download). Costs assume $0.70 to $0.90 an hour, 8 variants and 1 sample per prompt (`--n-samples 1`).

| Option | Prompts | Time per model | All three, incl. setup | Rough cost |
|---|---|---|---|---|
| 100 companies | 800 | ~8 min | ~1.25 hours | about $1 to $2 |
| 200 companies | 1,600 | ~16 min | ~1.5 hours | about $1 to $2 |
| 300 companies | 2,400 | ~25 min | ~2 hours | about $1.50 to $3 |
| 1,000 companies | 8,000 | ~80 min | ~5 hours | about $4 to $6 |
| 1,000 companies on an H100 (~$2.50 an hour) | 8,000 | ~35 min | ~2.5 hours | about $5 to $8 |

- The cost ranges allow for the run being up to about twice as slow as estimated. Add about $1 for the pilot and any debugging.
- An H100 saves time, not money. Rent one only if you want the results sooner.
- Samples of the same prompt add little independent information, so spend the budget on more companies, not more samples. At 5 samples per prompt, every row above becomes roughly 3 to 5 times longer.
- Larger companies (30 employees, 12 projects) give longer prompts and are slower than the 10-employee ones. A few 50-employee, 20-project companies can cost as much as dozens of small ones, so cap most companies at about 30 employees.
- Why it takes time even without training: each reply is generated one token at a time, and the long prompts limit how many replies fit in the card's memory at once.
- Your first-day RunPod spend, including a lot of debugging, was about $1.80.
- Your 3-prompt pilot, then a roughly 50-prompt timing run, gives the real numbers. Use those instead of this table.
