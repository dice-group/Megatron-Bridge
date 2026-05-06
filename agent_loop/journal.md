# JOURNAL — Variable-K Router Search

Append-only log. Each entry is one attempt: hypothesis (yours), the
mechanism (yours), the resulting val_loss curve (driver-recorded), and the
verdict.

## Format

Each entry is a `### iter NNNN` block with:
- **Hypothesis** — your one-line summary (driver copies from your reply)
- **Mechanism** — the journal note from your reply (one or two lines)
- **Val loss curve** — `iter:value` pairs, last two readings highlighted
- **Verdict** — `BETTER` / `WORSE` / `TIE` / `CRASHED` / `INVALID`
- **Delta vs baseline** — final val_loss minus baseline final val_loss

## Baseline

(Filled in by the driver before iter 0001 starts. If you see this header
without numbers below it, the baseline run is still pending.)

## Attempts

(Driver appends `### iter NNNN` blocks below this line.)
