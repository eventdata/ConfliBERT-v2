# Running the pipeline on NCSA Delta

This guide takes you from a fresh Delta account to a finished DAPT run. Everything
here also applies, with path changes, to any Slurm cluster.

## 0. What you need

- A Delta allocation (you know your short code, e.g. `bcaq`). GPU jobs bill the
  `<alloc>-delta-gpu` account, CPU jobs `<alloc>-delta-cpu`.
- This repo cloned to `/projects/<alloc>/$USER/conflibert-v2`.
- The Parquet corpus staged on scratch (see step 2).

**Filesystem rules (important):**

| Location | Use for | Caveat |
|---|---|---|
| `$HOME` | dotfiles only | tiny quota; do NOT put the HF cache here |
| `/projects/<alloc>` | code, venv, finished models | persistent, backed up |
| `/scratch/<alloc>` | corpus, packed tokens, checkpoints, HF cache | **purged after ~30 days idle** |

Anything on `/scratch` must be regenerable from code or synced elsewhere.
The path contract in `hpc/paths.env` encodes this split - do not fight it.

## 1. One-time setup (login node)

```bash
cd /projects/<alloc>/$USER
git clone <this-repo-url> conflibert-v2 && cd conflibert-v2
cp hpc/paths.env.example hpc/paths.env
# edit hpc/paths.env: set CB2_ALLOC to your allocation short code
bash hpc/setup_delta.sh
```

This builds the venv on `/projects`, installs pinned dependencies, pre-caches
`answerdotai/ModernBERT-base` into `$HF_HOME` on scratch, and smoke-tests imports.

## 2. Stage the corpus

The training input is the **Stage-1 Parquet corpus** (see `docs/PIPELINE.md` for
the schema). It is not in git. Either:

- copy it from the team share / Globus endpoint to `$CB2_CORPUS`
  (`/scratch/<alloc>/$USER/cb2/corpus/parquet`), preserving the
  one-subdirectory-per-source layout, or
- rebuild it from raw sources with `src/data/extract_corpus.py` (slow; only if
  you are changing the corpus itself), or
- ingest a NEW team source (any format: JSONL, CSV, TXT, converted PDFs) with
  `src/data/ingest_source.py`, following the step-by-step in
  `docs/PIPELINE.md` (register the source name first).

Sanity-check it before burning GPU hours:

```bash
source "$CB2_VENV/bin/activate"
python src/data/corpus_stats.py --corpus "$CB2_CORPUS" --out analysis/data
```

## 3. Pack tokens (CPU job - never do this on a GPU allocation)

```bash
bash hpc/submit.sh hpc/pack_corpus.sbatch
```

Writes `train_<seqlen>_hpc/` and `eval_random_<seqlen>_hpc/` under `$CB2_PACKED`.
Each contains `tokens.u16`, `segment_lengths.u16`,
`block_segment_offsets.u64`, and `meta.json`. Set `CB2_MAX_TOKENS` to cap the
budget (pilot runs used 2.5B and 5B); the default packs the full weighted pool.
The final selected document is always retained completely, so its tokens may
put the result slightly above the requested cap.
Check the tail of `logs/pack-*.out` for the `PACK_DONE` line and confirm
`n_tokens` in `meta.json` is what you intended.

## 4. Pretrain (GPU job)

```bash
bash hpc/submit.sh hpc/pretrain.sbatch
```

Defaults reproduce the pilot **v2-wsd recipe** on one A100x4 node: WSD scheduler,
peak LR 2e-4, MLM 0.30, seqlen 1024, global batch 256 blocks (~262k tokens/step),
one pass over the pack. Override via environment variables at submit time:

```bash
CB2_RUN=v2-hpc-10b CB2_LR=2e-4 CB2_SCHED=wsd bash hpc/submit.sh hpc/pretrain.sbatch
```

**ModernBERT-large.** Both model sizes are in scope and share one tokenizer,
so the same packs feed both. A large run is two overrides:

```bash
CB2_RUN=conflibert-v2-large CB2_BASE=answerdotai/ModernBERT-large CB2_BSZ=8 \
  bash hpc/submit.sh hpc/pretrain.sbatch
```

Halving `CB2_BSZ` keeps large within A100-40G memory at seqlen 1024; the
script recomputes accumulation so the global batch (and the LR recipe) is
unchanged. Treat the first large run as a shakedown: the pilot validated the
recipe on base only.

**Wall clock and resume.** The job requests 24 h. If training does not finish,
the checkpoint cadence (`--save-steps 500`) plus `--auto-resume` means you just
**resubmit the same command** and it continues from the last checkpoint. Nothing
else to do. This also covers preemption/requeue.

**Scaling notes.**

- `--bsz`/`--accum` are per-device; the script computes `accum` from
  `CB2_GLOBAL` so the global batch (and thus the LR recipe) is unchanged
  regardless of GPU count. If you change the global batch, retune the LR.
- Multi-node is not wired up (pilot evidence: at 150M params one A100 node is
  compute-sufficient; data volume, not FLOPs, is the binding constraint).
  If you need it, `torchrun --standalone` becomes `srun torchrun --rdzv...` -
  talk to the group first.
- OOM at `CB2_BSZ=16`? Drop to 8 (accum doubles automatically via the formula)
  or add `--grad-checkpointing` to the torchrun line.

**Monitoring.**

```bash
squeue -u $USER                       # queue state
tail -f logs/dapt-<jobid>.out         # live loss/step lines
# metrics.csv inside the run dir has step/tokens/loss/lr/tokens-per-sec;
# TensorBoard event files land in $CB2_SCRATCH/tb/<run-name>
```

## 5. Evaluate

```bash
bash hpc/submit.sh hpc/eval.sbatch "$CB2_MODELS/<run-name>" <ModelName>
```

Runs pseudo-perplexity vs the base model, the 7-task classification benchmark,
and the 2-task NER benchmark under the corrected protocol (prefix-space fix +
extended LR grid - see `docs/STATUS_AND_NEXT.md` for why this matters; the old
protocol was biased against ModernBERT-family models). Results append to CSVs in
`analysis/data/`. Requires the benchmark datasets (see `docs/PIPELINE.md`,
"External data").

## 6. Preserve the result

`/scratch` is purged. When a run is final:

```bash
rsync -av "$CB2_MODELS/<run-name>/" "/projects/<alloc>/$USER/models/<run-name>/"
# and/or push to the Hub:
hf auth login   # once
hf upload <org-or-user>/<run-name> "$CB2_MODELS/<run-name>"
```

Commit the run's `metrics.csv` + `run_summary.json` to `analysis/data/` (small,
and they are the provenance of every figure).

## Known pitfalls carried over from the pilot

- **Memmaps hate slow filesystems.** Packed token dirs must live on the fast
  local filesystem (`/scratch` Lustre is fine; the WSL pilot equivalent was
  "ext4, never 9p"). Do not train from packs on `$HOME`.
- **The WSD schedule needs the true total step count.** `train_dapt.py` computes
  it from pack size, batch geometry, and world size - if you cap with
  `--max-steps`, the decay tail is derived from that cap instead.
- **Do not eval with a mismatched tokenizer.** Pseudo-PPL comparisons are only
  valid between models sharing a tokenizer (`--tokenizer` flag pins it).
- **NER fine-tunes need `--prefix-space`** on ByteLevel-BPE models or scores are
  silently ~5 F1 too low. `hpc/eval.sbatch` already passes it.
