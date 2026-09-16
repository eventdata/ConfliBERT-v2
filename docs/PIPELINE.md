# Data pipeline contract

This document is the team agreement on what each stage consumes and produces.
If you change a format, a path convention, or a default here, you are changing
an interface other people's jobs depend on - update this file in the same PR.

## THE RULES (read these before touching anything)

1. **Data never enters git.** Corpora, packs, models, logs: they live on the
   cluster or the team share. Git holds code, docs, and results CSVs only.
2. **Every corpus source lands in the same Parquet schema** (below), one
   subdirectory per source, no matter what format it arrived in. No exceptions,
   no "temporary" side formats.
3. **Register a source name before first use** (step-by-step below). Source
   names are permanent; downstream weights and stats key on them.
4. **Model names are permanent and unique.** Results CSVs key on the name
   string; never reuse one for different weights.
5. **Results CSVs are append-only and committed.** Never hand-edit; delete bad
   rows in a commit that says why.
6. **Secrets stay in environment variables.** Never in files, never in git.
7. **Pack on CPU partitions, train on GPU partitions.** Never burn GPU
   allocation on tokenization.

```
raw archives ──► Stage 1 ──► Parquet corpus ──► Stage 2 ──► packed blocks ──► Stage 3 ──► model ──► Stage 4 ──► results CSVs
              extract_corpus            pack_tokens                train_dapt              eval harnesses
```

Filesystem note: **code paths are relative to the repo root; data paths are
machine-specific and flow in via CLI args** (locally) **or `hpc/paths.env`**
(on Delta). Never hardcode an absolute path in a script - thread it through an
argument or a `CB2_*` variable.

## Stage 1 - corpus construction (any input format -> one Parquet contract)

The full ConfliBERT-2 corpus combines sources from several teams (the original
ConfliBERT collection, Arizona's journal and dissertation data, WVU's
FineWeb-derived and scraped text). Teams bring different file formats; the
contract is that **everything converges to one Parquet schema**, laid out as
`<corpus-out>/parquet/<Source>/<shard>.parquet`. Downstream stages infer the
source from that **parent directory name** - do not flatten the layout.

Row schema (all stages depend on these columns):

| column | meaning |
|---|---|
| `id` | stable document id (`<Source>:<file>:<record>`) |
| `source` | source name, same as the folder |
| `date`, `year` | publication date (`year = -1` if unknown) |
| `n_chars`, `n_words` | length metadata |
| `text` | ftfy-cleaned document text |
| `text_hash` | exact-dedup key (blake2b-8 of the cleaned text) |
| `split` | `train` \| `eval_random` (0.3% random) \| `eval_temporal` (year 2021) |

Splits are assigned deterministically from the text hash and date, identically
across all sources and ingest scripts, so held-out evaluation stays valid no
matter who ingested what.

### Adding a new corpus source, step by step

1. **Register the name.** Pick one CamelCase source name (e.g.
   `ArizonaJournals`) and add it to the source registry below in the same PR
   that adds the data pipeline for it. Names are permanent.
2. **Get the raw files into a text interchange format.** Supported directly:
   `.jsonl`/`.jsonl.gz` (preferred; one JSON object per line), `.json` (array),
   `.csv`/`.tsv` (header row required), `.txt`/`.txt.gz` (one document per
   file). Anything binary (PDF, DOCX, WARC) gets a small source-specific
   converter to JSONL first; `src/data/crisiswatch_parse.py` is a worked PDF
   example and `pypdf` is already a dependency.
3. **Ingest.** `src/data/ingest_source.py` applies the same cleaning, hashing,
   and split logic as the original extractor, whatever the input format:

   ```bash
   python src/data/ingest_source.py \
     --source ArizonaJournals \
     --input /path/to/raw/dump \
     --out "$CB2_SCRATCH/corpus" \
     --text-field body --date-field published --id-field doi
   ```

   The field flags map your column/key names onto the schema. It refuses to
   run into a non-empty source dir (`--append` to continue an interrupted
   ingest). Documents under 32 characters are dropped.
4. **Validate before anyone packs.**
   `python src/data/corpus_stats.py --corpus <out>/parquet --out analysis/data`
   and check: document count and total characters match expectations, all
   three splits are present, and the per-year table looks sane.
5. **Spot-read a sample.** Open a few rows and look at `text` for mojibake,
   boilerplate, HTML debris, or truncation. Encoding problems found after
   packing cost a repack; found here they cost a rerun of one script.
6. **Commit the paper trail.** The updated `analysis/data/corpus_stats.csv`,
   the registry row, and any converter script go in the PR. The data itself
   never does.
7. **Agree on a source weight.** Packing keeps `weight` fraction of each
   source (see Stage 2). A new source is invisible to weighted packs until the
   team adds it to the standard weights, so raise it in the group before the
   next pack is built.

### Source registry

| source | origin | ingested by |
|---|---|---|
| `News`, `Organization`, `Gigaword`, `UTDstory`, `Wikipedia` | original EN-Politics archives (`.json.tar.gz`) | `extract_corpus.py` |
| Arizona journals + dissertations | planned | `ingest_source.py` (name TBD, register here) |
| WVU FineWeb + scraped text | planned | `ingest_source.py` (name TBD, register here) |

The original archives live at
`<corpus-root>/1945-2021 json_files_with_metadata/` on the team share; they are
processed with `python src/data/extract_corpus.py --corpus-root <raw-root>
--out <corpus-out>` (resumable; skips shards whose output exists). Ask the team
for the share/Globus location; do not re-download sources independently.

## Stage 2 - packed token blocks (`src/data/pack_tokens.py`)

```bash
python src/data/pack_tokens.py --corpus .../parquet --split train \
  --seqlen 1024 --source-weights "News=1.0,Organization=1.0,UTDstory=1.0,Gigaword=0.7,Wikipedia=0.25" \
  --max-tokens <budget> --dedup --seed 7 --out <packed>/train_1024_<tag>
```

Documents are tokenized without truncation. A document stays intact when its
content plus the tokenizer's `[CLS]` and `[SEP]` tokens fit within `seqlen`;
only longer documents are split, and no content tokens are dropped. Every
resulting segment is stored as `[CLS] content [SEP]` and packed with an online,
bounded-memory best-fit algorithm.

**Output contract** - a directory containing four files:

- `tokens.u16` - uint16 memmap with shape `(n_blocks, seqlen)`; unused block
  tails contain the tokenizer's PAD ID for fixed-width storage;
- `segment_lengths.u16` - the real length of every document segment;
- `block_segment_offsets.u64` - offsets mapping each block to its range in
  `segment_lengths.u16`;
- `meta.json` - format version, dimensions, real/storage token counts, packing
  efficiency, tokenizer IDs, and provenance. Never edit it by hand or mix packs
  created with different tokenizers.

`--max-tokens` counts real tokens, including `[CLS]` and `[SEP]`. Once a
document has been selected it is kept in full, so the cap may be exceeded by
that final document rather than truncating it.

**Naming convention:** `train_<seqlen>_<tag>` / `eval_random_<seqlen>_<tag>`.
The `<tag>` says what is inside (`native`, `aug`, `5b`, `hpc`...). Packs must
live on fast local storage (Lustre `/scratch` on Delta; ext4 in the WSL pilot -
never a network/9p mount, memmap random reads will crawl).

**Standard source weights** (used by every headline run): `News=1.0,
Organization=1.0, UTDstory=1.0, Gigaword=0.7, Wikipedia=0.25`. Changing them is
an experiment, not a default - tag the pack accordingly.

## Stage 3 - DAPT (`src/pretrain/train_dapt.py`)

Single GPU and `torchrun` DDP both work; see `hpc/pretrain.sbatch` for the
canonical Delta invocation and `docs/HPC_DELTA.md` for scaling rules.
The collator removes stored PAD tails and uses each segment length as a
FlashAttention boundary. Hugging Face then flattens the batch, resets positions
at every segment, and prevents attention between documents.
`torch.compile` is enabled by default through ModernBERT's selective
`reference_compile` path: embeddings, MLPs, and the MLM head are compiled while
FlashAttention 2 stays outside the compiled graph. Full-model compilation is
avoided because it can cause an FP32/BF16 dtype mismatch around FlashAttention.
Use `--no-torch-compile` to disable this selective compilation.

**FlashAttention installation.** When a CUDA compiler is available, prefer a
forced source build after installing PyTorch. This builds FlashAttention
against the installed PyTorch/CUDA environment and reduces the risk of ABI or
undefined-symbol errors from an incompatible prebuilt wheel:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
  python -m pip install "flash-attn==2.6.3" \
  --no-build-isolation --no-cache-dir
```

**Two model sizes are in scope for the scaled run**: `answerdotai/ModernBERT-base`
(150M params) and `answerdotai/ModernBERT-large` (395M). They share one
tokenizer, so Stage-2 packs are built once and reused for both; a size is just
`--base` plus a smaller per-device batch (large: halve `CB2_BSZ`; the launcher
recomputes accumulation to keep the global batch identical). The pilot recipes
below were validated on base; treat the first large run as a shakedown, not a
production run.

**Reference recipes** (pilot-validated; global batch 256 blocks ≈ 262k tokens/step):

| recipe | base | scheduler | peak LR | notes |
|---|---|---|---|---|
| native (A2) | `answerdotai/ModernBERT-base` | cosine | 5e-5 | simplest baseline |
| **wsd (A3, headline)** | pre-decay stable ckpt via `convert_stable_ckpt.py` | wsd | 2e-4 | warmup 0.03, decay tail 0.20 |
| aug-wsd (A6) | stable ckpt + `augment_model.py --rescale-norms` | wsd | 2e-4 | + `--new-embed-warmup-steps 400` |

The WSD recipes need the ModernBERT **pre-decay stable checkpoint** (a Composer
`.pt`, not in git - team share) converted once with
`src/pretrain/convert_stable_ckpt.py`.

**Output contract** - the run dir contains an HF model (`config.json`,
`model.safetensors`, tokenizer files), rolling `checkpoint-*` dirs (deleted on
finalize), plus two files you must preserve with every run:

- `metrics.csv` - step, tokens_seen, losses, lr, tokens/sec (feeds the R figures);
- `run_summary.json` - full args + tokens seen + GPU hours (the run's provenance).

Finalized models are named `conflibert-v2-<recipe>[-<budget>]` and keep only the
model + tokenizer + the two provenance files.

## Stage 4 - evaluation

**Two intrinsic rules that are easy to get wrong:**

1. Pseudo-PPL (`src/eval/pseudo_ppl.py`) is only comparable **between models
   sharing a tokenizer**; pin it with `--tokenizer`.
2. NER fine-tunes on ByteLevel-BPE models require `--prefix-space`, and the LR
   grid must extend to 2.4e-4. Without both, ModernBERT-family scores are
   silently depressed (this bug cost us a wrong conclusion once - see
   `docs/STATUS_AND_NEXT.md`, 2026-08-06).

**Benchmark protocol** (`scripts/corrected_bench.py`, wraps
`src/eval/finetune_benchmark.py`): per (task, model), sweep the LR grid with
seed 123 and dev selection, then run seeds 124/125 at the best-dev LR. Nine
tasks total (7 classification + 2 NER) drawn from `external/ConfliBERT/data`.

**Results contract** - every eval harness **appends** rows to a CSV under
`analysis/data/`, keyed by (model name, task, seed, lr). These CSVs are
committed to git: they are the experimental record, and every figure and table
regenerates from them. Consequences:

- pick a **globally unique model name** before running (`ConfliBERT-v2-<recipe>`;
  suffix `-psfix` for prefix-space NER rows) - collisions corrupt the record;
- reruns append: if a job died mid-config, **dedupe by key before analysis**
  (keep-last), or delete the partial rows first;
- never edit a committed results CSV by hand; if a run was wrong, delete its
  rows in a commit whose message says why.

Paper experiment harnesses (`ipe_crossfit`, `tsv_crossfit`, `win_crossfit`,
`cw_finetune`, `llm_*_baseline`, `throughput_bench`) follow the same
append-to-`analysis/data` contract. LLM baselines read the API key from
`$OPENROUTER_API_KEY` (preferred) or a local `.openrouter_key` file - which is
gitignored and must stay that way.

## External data (not in git - fetch on setup)

| path | what | how to get it |
|---|---|---|
| `external/ConfliBERT/` | benchmark tasks + configs | `git clone https://github.com/eventdata/ConfliBERT external/ConfliBERT` |
| `external/IndiaPoliceEvents/` | IPE corpus (E1/E2/E4/E5) | `git clone https://github.com/slanglab/IndiaPoliceEvents external/IndiaPoliceEvents` |
| `external/crisiswatch/` | CrisisWatch PDFs + derived task | team share, or rebuild: `crisiswatch_download.py` → `crisiswatch_parse.py` → `crisiswatch_task.py` (downloader needs internet + `curl`; run it on a login node, be polite: it rate-limits itself) |
| ModernBERT stable ckpt (`.pt`) | WSD recipes' starting point | team share; convert with `convert_stable_ckpt.py` |
| raw corpus archives | Stage 0 | team share / Globus |

## Known warts (fix welcome, coordinate first)

- `src/data/crisiswatch_task.py` has no CLI args; run it from the repo root.
- `src/eval/throughput_bench.py` hardcodes its model list in `CONFIGS`; edit
  before running on new models, and re-run on the target hardware (timings are
  hardware-specific).
- `analysis/*.R` figure scripts still `setwd()` to the pilot Windows path;
  results CSVs are portable, the R scripts are not yet.
- `scripts/run_queue*.sh` chain jobs by polling log files for sentinel strings,
  pilot-rig style. On Delta use `sbatch --dependency=afterok:<jobid>` instead.
