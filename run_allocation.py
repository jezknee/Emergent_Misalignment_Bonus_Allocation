"""Run the bonus-allocation prompts through one model with vLLM and save the responses.

Usage (on the GPU pod, one model per process so GPU memory is freed between models):

    # pilot: first 3 prompts, to check the whole pipeline
    python run_allocation.py --prompts prompts.jsonl --model unsloth/Qwen2.5-Coder-7B-Instruct \\
        --n-samples 2 --limit 3 --out pilot_base.jsonl

    # full run, safe against dropped connections
    nohup python run_allocation.py --prompts prompts.jsonl \\
        --model YOUR_HF_USERNAME/qwen-coder-7b-insecure > run_insecure.log 2>&1 &

Output is a JSONL file with one row per generated sample (prompt_id, sample_idx,
model, response, finish_reason, token counts, ...). Results are written after every
chunk of prompts, and rerunning the same command resumes where it stopped: prompts
that already have all their samples are skipped, and incomplete ones are redone.
Private models need HF_TOKEN set. Join the responses back to the prompt table with
`merge_with_prompts`.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------- loading and joining
def load_prompts(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    return pd.read_csv(path)


def load_responses(path):
    """Read a responses file, skipping any half-written line and duplicate samples."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["prompt_id", "sample_idx", "model"], keep="last")


def merge_with_prompts(prompts_path, responses_path):
    """One row per generated sample: the prompt table's columns plus the response
    columns. Like the emergent-misalignment table, with `response` as its own column."""
    prompts = load_prompts(prompts_path).drop(columns=["response"], errors="ignore")
    overlap = [c for c in ("variant", "n_employees", "n_projects") if c in prompts.columns]
    responses = load_responses(responses_path).drop(columns=overlap + ["company_seed"],
                                                    errors="ignore")
    return responses.merge(prompts, on="prompt_id", how="left")


# ------------------------------------------------------------------------- helpers
def to_messages(row):
    """Chat messages for a prompt row. An empty system message is omitted so the
    Qwen chat template inserts its default one, as in the training runs."""
    messages = []
    system = row.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": row["user"]})
    return messages


def count_prompt_tokens(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def load_existing(out_path, model, n_samples, force=False):
    """Keep only complete prompts from an earlier run and return their ids.

    Rewrites the file without incomplete prompts or half-written lines, so resuming
    never leaves duplicates behind.
    """
    if not out_path.exists():
        return set()
    rows = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line cut off by a crash
    other = {r["model"] for r in rows} - {model}
    if other and not force:
        raise SystemExit(f"{out_path} already holds results from {sorted(other)}, not {model}. "
                         "Use a different --out, or --force to keep both in one file.")
    counts = {}
    for r in rows:
        if r["model"] == model:      # another model's samples never count as done
            counts.setdefault(r["prompt_id"], set()).add(r["sample_idx"])
    done = {pid for pid, idx in counts.items() if len(idx) >= n_samples}
    keep = [r for r in rows if r["prompt_id"] in done or r["model"] != model]
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    tmp.replace(out_path)
    return done


def _build_vllm(args, max_model_len):
    from vllm import LLM
    return LLM(model=args.model, max_model_len=max_model_len, seed=args.seed,
               gpu_memory_utilization=args.gpu_memory_utilization)


# ----------------------------------------------------------------------------- run
def run(args, llm=None, tokenizer=None, sampling_params_cls=None):
    """Generate and save responses. Returns (out_path, stats).

    llm, tokenizer and sampling_params_cls can be passed in (used for testing
    without a GPU); otherwise they are created from vLLM and transformers.
    """
    prompts = load_prompts(args.prompts)
    if args.variants:
        keep = set(args.variants.split(","))
        available = sorted(prompts["variant"].unique())
        prompts = prompts[prompts["variant"].isin(keep)]
        if prompts.empty:
            raise SystemExit(f"No prompts match --variants {args.variants!r}. "
                             f"Variants in the table: {', '.join(available)}")
    if args.limit:
        prompts = prompts.head(args.limit)
    if prompts.empty:
        raise SystemExit("No prompts selected.")

    out_path = Path(args.out or f"responses_{args.model.rstrip('/').split('/')[-1]}.jsonl")
    done = load_existing(out_path, args.model, args.n_samples, args.force)

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = prompts.to_dict("records")
    lengths = [count_prompt_tokens(tokenizer, to_messages(r)) for r in records]
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    needed = lengths[longest] + args.max_tokens
    needed = -(-needed // 512) * 512  # round up to a multiple of 512
    if args.max_model_len and args.max_model_len < needed:
        raise SystemExit(
            f"--max-model-len {args.max_model_len} is too small: the longest prompt "
            f"({records[longest]['prompt_id']}) is {lengths[longest]} tokens and replies can "
            f"use {args.max_tokens} more, so you need at least {needed}.")
    max_model_len = args.max_model_len or needed

    todo = [r for r in records if r["prompt_id"] not in done]
    print(f"{len(records)} prompts selected, {len(done)} already complete, {len(todo)} to run. "
          f"Longest prompt {lengths[longest]} tokens, max_model_len {max_model_len}.", flush=True)

    stats = {"prompts_run": len(todo), "completions": 0, "truncated": 0,
             "completion_tokens": 0, "max_model_len": max_model_len,
             "longest_prompt_tokens": lengths[longest]}
    if not todo:
        return out_path, stats

    if llm is None:
        llm = _build_vllm(args, max_model_len)
    if sampling_params_cls is None:
        from vllm import SamplingParams as sampling_params_cls
    params = sampling_params_cls(n=args.n_samples, temperature=args.temperature,
                                 top_p=args.top_p, max_tokens=args.max_tokens)

    started = time.time()
    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        t0 = time.time()
        outputs = llm.chat([to_messages(r) for r in chunk], params, use_tqdm=False)
        lines = []
        for row, out in zip(chunk, outputs):
            for k, comp in enumerate(out.outputs):
                n_tokens = len(comp.token_ids)
                stats["completions"] += 1
                stats["completion_tokens"] += n_tokens
                stats["truncated"] += comp.finish_reason == "length"
                lines.append(json.dumps({
                    "prompt_id": row["prompt_id"], "sample_idx": k, "model": args.model,
                    "response": comp.text, "finish_reason": comp.finish_reason,
                    "prompt_tokens": len(out.prompt_token_ids), "completion_tokens": n_tokens,
                    "temperature": args.temperature, "top_p": args.top_p,
                    "max_tokens": args.max_tokens, "sampling_seed": args.seed,
                    "company_seed": row.get("seed"), "variant": row.get("variant"),
                    "n_employees": row.get("n_employees"), "n_projects": row.get("n_projects"),
                }))
        with open(out_path, "a", encoding="utf-8") as f:  # saved after every chunk
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finished = start + len(chunk)
        print(f"chunk {start // args.chunk_size + 1}: prompts {finished}/{len(todo)} "
              f"({time.time() - t0:.0f}s, {time.time() - started:.0f}s total)", flush=True)

    if stats["completions"]:
        print(f"Done: {stats['completions']} completions, "
              f"{stats['truncated']} cut off at --max-tokens "
              f"({100 * stats['truncated'] / stats['completions']:.1f}%), "
              f"mean {stats['completion_tokens'] / stats['completions']:.0f} tokens. "
              f"Saved to {out_path}", flush=True)
    return out_path, stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prompts", required=True, help="prompt table (.jsonl or .csv)")
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--out", help="responses file (default: responses_<model name>.jsonl)")
    p.add_argument("--n-samples", type=int, default=5, help="samples per prompt")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2048, help="longest reply allowed")
    p.add_argument("--max-model-len", type=int,
                   help="context length for vLLM (default: longest prompt plus --max-tokens)")
    p.add_argument("--chunk-size", type=int, default=20, help="prompts per saved chunk")
    p.add_argument("--limit", type=int, help="only the first N prompts (for a pilot)")
    p.add_argument("--variants", help="comma-separated variants to run, e.g. assistant_fair_direct,analyst_fair_vote")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--force", action="store_true",
                   help="allow a responses file that already holds another model's results")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
