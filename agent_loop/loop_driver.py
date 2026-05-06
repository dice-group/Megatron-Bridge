"""Agent loop driver — searches for a better variable-K MoE router.

Per iteration:
  1. Send program.md + journal.md + current TopAnyRouter source to the
     remote LLM agent.
  2. Parse the agent's reply: hypothesis, code, journal note.
  3. Validate the new code (ast.parse + invariants), splice into gate.py.
  4. Run training locally for ~20 min via scripts/local_train_super_small.sh.
  5. Parse val_loss from stdout, compare to baseline, append journal entry.
  6. Revert gate.py to best-so-far if the iteration didn't improve.

Bootstrapping: if baseline.json is missing, run topk first, cache curve.

Usage:
    python agent_loop/loop_driver.py --max-iters 30
    python agent_loop/loop_driver.py --skip-baseline   # if baseline.json exists
"""

import argparse
import ast
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


REPO = Path(__file__).resolve().parent.parent
LOOP_DIR = REPO / "agent_loop"
GATE_PY = REPO / "3rdparty/Megatron-LM/megatron/core/transformer/moe/gate.py"
TRAIN_SH = REPO / "scripts/local_train_super_small.sh"
PROGRAM_MD = LOOP_DIR / "program.md"
JOURNAL_MD = LOOP_DIR / "journal.md"
BASELINE_JSON = LOOP_DIR / "baseline.json"
LAST_GOOD = LOOP_DIR / "last_good_gate.py"
HISTORY = LOOP_DIR / "history"

API_URL = os.environ.get("AGENT_API_URL",
                         "https://dice-llm-chat.cs.uni-paderborn.de/api/v1/chat/completions")
API_KEY = os.environ.get("AGENT_API_KEY")
MODEL = os.environ.get("AGENT_MODEL", "nemotron-3-super-120B-a12b")

VAL_LOSS_RE = re.compile(
    r"validation loss at iteration (\d+).*?\| lm_loss value: ([\d.E+\-]+) \|"
)


# ─── Code splicing (AST-based to avoid swallowing module-level helpers) ────

def extract_class_source(source: str, class_name: str) -> tuple[int, int]:
    """Return (start_offset, end_offset) of `class {class_name}` block.

    Uses ast so we don't accidentally include module-level helpers
    (e.g. `_sweep_diag_log`) that sit between sibling classes.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            lines = source.splitlines(keepends=True)
            # ast lineno/end_lineno are 1-indexed; end_lineno is inclusive.
            start = sum(len(line) for line in lines[: node.lineno - 1])
            end = sum(len(line) for line in lines[: node.end_lineno])
            return start, end
    raise ValueError(f"class {class_name} not found in source")


def splice_class(source: str, class_name: str, new_class_src: str) -> str:
    start, end = extract_class_source(source, class_name)
    # Preserve exactly one trailing blank line between classes.
    new_block = new_class_src.rstrip() + "\n\n\n"
    return source[:start] + new_block + source[end:]


def get_class_source(source: str, class_name: str) -> str:
    start, end = extract_class_source(source, class_name)
    return source[start:end]


# ─── Code validation ────────────────────────────────────────────────────────

FORBIDDEN_PATTERNS = [
    (r"\btorch\.topk\b", "torch.topk is forbidden — variable-K must not select fixed top-K"),
    (r"\bF\.topk\b", "F.topk is forbidden — variable-K must not select fixed top-K"),
]

REQUIRED_IN_CLASS = [
    (r"class TopAnyRouter\(Router\):", "must keep `class TopAnyRouter(Router):`"),
    (r"def forward\(", "must keep a `forward` method"),
    (r"def routing\(", "must keep a `routing` method"),
]


def validate_new_class(class_src: str) -> tuple[bool, str]:
    """Return (ok, error_message). Validates the agent's proposed class body."""
    for pat, msg in FORBIDDEN_PATTERNS:
        if re.search(pat, class_src):
            return False, f"FORBIDDEN: {msg}"
    for pat, msg in REQUIRED_IN_CLASS:
        if not re.search(pat, class_src):
            return False, f"MISSING: {msg}"
    return True, ""


def validate_full_file(file_path: Path) -> tuple[bool, str]:
    """Try to parse the whole gate.py — catches splice mistakes."""
    try:
        src = file_path.read_text()
        ast.parse(src)
    except SyntaxError as e:
        return False, f"SYNTAX ERROR: {e}"
    return True, ""


# ─── Agent reply parsing ────────────────────────────────────────────────────

def parse_agent_reply(text: str) -> dict:
    """Extract hypothesis, code, journal note from the agent's markdown reply."""
    out = {"hypothesis": "", "code": "", "journal_note": "", "raw": text}

    def section(name: str, src: str) -> str:
        m = re.search(
            rf"^##\s*{re.escape(name)}\s*$\n(.*?)(?=^##\s|\Z)",
            src,
            re.MULTILINE | re.DOTALL,
        )
        return m.group(1).strip() if m else ""

    out["hypothesis"] = section("Hypothesis", text)
    code_blob = section("Code", text)
    out["journal_note"] = section("Journal note", text)

    # Extract python code from the code section's fenced block.
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", code_blob, re.DOTALL)
    out["code"] = fence.group(1).strip() if fence else code_blob.strip()
    return out


# ─── API call ───────────────────────────────────────────────────────────────

def call_agent(system: str, user: str, max_retries: int = 3) -> str:
    if not API_KEY:
        raise RuntimeError("AGENT_API_KEY not set — add it to .env (see .env.example)")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(API_URL, json=payload, headers=headers, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            print(f"[api] attempt {attempt+1}/{max_retries} failed: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Agent API failed after {max_retries} retries: {last_err}")


# ─── Training run ───────────────────────────────────────────────────────────

def run_training(routing_type: str, log_path: Path, run_name: str, timeout_sec: int) -> int:
    """Launch local_train_super_small.sh with timeout. Returns exit code."""
    env = os.environ.copy()
    env["RUN_NAME"] = run_name
    env["ROUTING_TYPE"] = routing_type
    # K-target hyperparams locked per program.md.
    env["AUX_LOSS_COEFF"] = "0.01"
    env["TOPANY_K_TARGET"] = "2.5"
    env["TOPANY_K_TARGET_COEFF"] = "0.3"
    env["TOPANY_FORCE_TOP1"] = "1"
    # `timeout --signal=INT` lets the trainer flush logs before exit.
    cmd = [
        "timeout", "--signal=INT", "--kill-after=60",
        f"{timeout_sec}", "bash", str(TRAIN_SH),
    ]
    print(f"[train] {' '.join(cmd)}  (log → {log_path})", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def parse_val_curve(log_path: Path) -> list[tuple[int, float]]:
    """Return list of (iteration, val_loss) tuples from a training log."""
    text = log_path.read_text(errors="replace")
    return [(int(i), float(v)) for i, v in VAL_LOSS_RE.findall(text)]


# ─── Verdict ────────────────────────────────────────────────────────────────

def verdict(curve: list[tuple[int, float]], baseline: list[tuple[int, float]],
            margin: float = 0.02) -> str:
    if not curve:
        return "CRASHED"
    if not baseline:
        return "NO_BASELINE"
    # Compare final readings.
    last_val = curve[-1][1]
    base_last = baseline[-1][1]
    delta = last_val - base_last
    if delta < -margin:
        return "BETTER"
    if delta > margin:
        return "WORSE"
    return "TIE"


# ─── Journal ────────────────────────────────────────────────────────────────

def append_journal_entry(
    iter_n: int, hypothesis: str, note: str,
    curve: list[tuple[int, float]], baseline: list[tuple[int, float]],
    verdict_str: str, extra: str = "",
) -> None:
    base_last = baseline[-1][1] if baseline else None
    last_val = curve[-1][1] if curve else None
    delta = (last_val - base_last) if (last_val is not None and base_last is not None) else None

    curve_str = ", ".join(f"{i}:{v:.4f}" for i, v in curve) if curve else "(none)"
    delta_str = f"{delta:+.4f}" if delta is not None else "n/a"

    entry = (
        f"\n### iter {iter_n:04d}  ({dt.datetime.now().isoformat(timespec='seconds')})\n"
        f"- **Verdict**: `{verdict_str}`  **Δ vs baseline**: `{delta_str}`\n"
        f"- **Hypothesis**: {hypothesis.strip() or '(none)'}\n"
        f"- **Mechanism**: {note.strip() or '(none)'}\n"
        f"- **Val curve**: {curve_str}\n"
    )
    if extra:
        entry += f"- **Note**: {extra.strip()}\n"
    with open(JOURNAL_MD, "a") as f:
        f.write(entry)


def write_baseline_to_journal(curve: list[tuple[int, float]]) -> None:
    """Replace the `## Baseline` placeholder with the recorded curve."""
    text = JOURNAL_MD.read_text()
    curve_str = ", ".join(f"{i}:{v:.4f}" for i, v in curve)
    final = curve[-1][1] if curve else float("nan")
    block = (
        f"## Baseline\n\n"
        f"Run: `topk` (k=2). Final val lm_loss: **{final:.4f}**.\n\n"
        f"Curve: {curve_str}\n"
    )
    text = re.sub(r"## Baseline\n.*?(?=\n## )", block + "\n", text, count=1, flags=re.DOTALL)
    JOURNAL_MD.write_text(text)


# ─── Bootstrapping baseline ────────────────────────────────────────────────

def bootstrap_baseline(timeout_sec: int) -> list[tuple[int, float]]:
    if BASELINE_JSON.exists():
        data = json.loads(BASELINE_JSON.read_text())
        return [(int(i), float(v)) for i, v in data["curve"]]
    HISTORY.mkdir(parents=True, exist_ok=True)
    log_path = HISTORY / "baseline_topk.log"
    print("[baseline] running topk reference run...", flush=True)
    rc = run_training("topk", log_path, run_name="agent_loop_baseline_topk",
                      timeout_sec=timeout_sec)
    curve = parse_val_curve(log_path)
    print(f"[baseline] exit {rc}, parsed {len(curve)} val readings", flush=True)
    if not curve:
        print("[baseline] FATAL: no val_loss readings parsed. Inspect log:", log_path)
        sys.exit(1)
    BASELINE_JSON.write_text(json.dumps({
        "curve": curve,
        "exit_code": rc,
        "log_path": str(log_path),
        "timestamp": dt.datetime.now().isoformat(),
    }, indent=2))
    write_baseline_to_journal(curve)
    return curve


# ─── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--train-timeout", type=int, default=1200,
                    help="Per-iteration training wall time in seconds (default 1200 = 20 min)")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip baseline; require baseline.json to exist")
    ap.add_argument("--start-iter", type=int, default=1,
                    help="Iteration index to start at (resume runs)")
    args = ap.parse_args()

    HISTORY.mkdir(parents=True, exist_ok=True)

    # Preflight: train script + container + data exist
    if not TRAIN_SH.exists():
        sys.exit(f"missing {TRAIN_SH}")
    if not GATE_PY.exists():
        sys.exit(f"missing {GATE_PY}")

    # Snapshot the current gate.py as the initial last-good (baseline state).
    if not LAST_GOOD.exists():
        shutil.copy(GATE_PY, LAST_GOOD)
        print(f"[init] snapshotted current gate.py → {LAST_GOOD}", flush=True)

    # Baseline
    if args.skip_baseline:
        if not BASELINE_JSON.exists():
            sys.exit("--skip-baseline set but baseline.json missing")
        baseline = [(int(i), float(v)) for i, v in
                    json.loads(BASELINE_JSON.read_text())["curve"]]
        print(f"[baseline] cached: final={baseline[-1][1]:.4f}", flush=True)
    else:
        baseline = bootstrap_baseline(timeout_sec=args.train_timeout)
        # Restore gate.py to last-good after baseline (topk doesn't touch gate.py
        # for routing, but be explicit).
        shutil.copy(LAST_GOOD, GATE_PY)

    # Best-so-far is whatever last_good_gate.py has (initially = current state).
    best_curve = baseline
    best_final = baseline[-1][1]

    program_text = PROGRAM_MD.read_text()

    for it in range(args.start_iter, args.max_iters + 1):
        iter_dir = HISTORY / f"iter_{it:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*72}\n[iter {it:04d}] starting\n{'='*72}", flush=True)

        # Always show the agent the *best-so-far* TopAnyRouter, not the latest tried.
        shutil.copy(LAST_GOOD, GATE_PY)
        current_class_src = get_class_source(GATE_PY.read_text(), "TopAnyRouter")
        journal_text = JOURNAL_MD.read_text()

        user_msg = (
            f"# Current journal\n\n{journal_text}\n\n"
            f"---\n\n"
            f"# Current TopAnyRouter source (best-so-far)\n\n"
            f"```python\n{current_class_src}\n```\n\n"
            f"---\n\n"
            f"Propose your next variant. Reply with the three sections "
            f"`## Hypothesis`, `## Code`, `## Journal note` as specified."
        )

        # Call agent
        try:
            reply = call_agent(program_text, user_msg)
        except Exception as e:
            append_journal_entry(it, "(api failure)", "(no reply)", [], baseline,
                                 "API_FAIL", extra=str(e))
            continue
        (iter_dir / "agent_reply.md").write_text(reply)
        parsed = parse_agent_reply(reply)
        if not parsed["code"]:
            append_journal_entry(it, parsed["hypothesis"], parsed["journal_note"],
                                 [], baseline, "INVALID",
                                 extra="agent reply contained no code block")
            continue

        # Validate proposed class
        ok, err = validate_new_class(parsed["code"])
        if not ok:
            append_journal_entry(it, parsed["hypothesis"], parsed["journal_note"],
                                 [], baseline, "INVALID", extra=err)
            (iter_dir / "rejected_class.py").write_text(parsed["code"])
            continue

        # Splice into gate.py
        try:
            new_full = splice_class(LAST_GOOD.read_text(), "TopAnyRouter", parsed["code"])
            GATE_PY.write_text(new_full)
        except Exception as e:
            append_journal_entry(it, parsed["hypothesis"], parsed["journal_note"],
                                 [], baseline, "INVALID", extra=f"splice failed: {e}")
            shutil.copy(LAST_GOOD, GATE_PY)
            continue

        ok, err = validate_full_file(GATE_PY)
        if not ok:
            append_journal_entry(it, parsed["hypothesis"], parsed["journal_note"],
                                 [], baseline, "INVALID", extra=err)
            shutil.copy(LAST_GOOD, GATE_PY)
            continue

        # Save snapshot of attempted gate.py BEFORE running it (so we have it
        # even if the training crashes hard).
        shutil.copy(GATE_PY, iter_dir / "gate.py")

        # Run training
        log_path = iter_dir / "train.log"
        run_name = f"agent_loop_iter_{it:04d}"
        rc = run_training("topany", log_path, run_name=run_name,
                          timeout_sec=args.train_timeout)
        curve = parse_val_curve(log_path)
        v = verdict(curve, baseline)
        print(f"[iter {it:04d}] rc={rc} verdict={v} curve={len(curve)} readings", flush=True)

        # If actually better, promote to last-good.
        if v == "BETTER" and curve and curve[-1][1] < best_final:
            shutil.copy(GATE_PY, LAST_GOOD)
            best_final = curve[-1][1]
            best_curve = curve
            print(f"[iter {it:04d}] new best: {best_final:.4f}", flush=True)
        else:
            # Revert.
            shutil.copy(LAST_GOOD, GATE_PY)

        append_journal_entry(it, parsed["hypothesis"], parsed["journal_note"],
                             curve, baseline, v,
                             extra=(f"exit_code={rc}" if v == "CRASHED" else ""))
        # Persist iteration result for resumability / inspection.
        (iter_dir / "result.json").write_text(json.dumps({
            "iter": it,
            "verdict": v,
            "exit_code": rc,
            "curve": curve,
            "best_final_so_far": best_final,
            "hypothesis": parsed["hypothesis"],
            "journal_note": parsed["journal_note"],
        }, indent=2))

    print(f"\n[done] best final val_loss: {best_final:.4f} "
          f"(baseline: {baseline[-1][1]:.4f})", flush=True)


if __name__ == "__main__":
    main()
