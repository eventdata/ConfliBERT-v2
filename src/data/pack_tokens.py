#!/usr/bin/env python
"""
Stage 2: tokenize + greedily pack documents into fixed-length MLM blocks.

Documents are kept whole unless they exceed `--seqlen`. Long documents are
split without dropping tokens. Every segment uses the tokenizer's standard
special-token template. Segment lengths are saved so the training code can
later prevent attention between documents and remove stored padding.
"""
from __future__ import annotations
import argparse, bisect, glob, json, os, random
import numpy as np

MODERNBERT = "answerdotai/ModernBERT-base"


def parse_weights(s: str):
    if not s:
        return None
    w = {}
    for part in s.split(","):
        k, v = part.split("=")
        w[k.strip()] = float(v)
    return w


def document_segments(token_ids, cls, sep, content_length):
    """Split long content and add the tokenizer's special tokens to each part."""
    chunks = [token_ids[i:i + content_length]
              for i in range(0, len(token_ids), content_length)]
    return [[cls] + chunk + [sep] for chunk in chunks]


class GreedyBestFitPacker: # very similar to what ModernBERT's GreedyBestFitSequencePacker
    """Place each document segment in the fullest block where it fits."""
    def __init__(self, seqlen, buffer_size, write_block):
        self.seqlen = seqlen
        self.buffer_size = buffer_size
        self.write_block = write_block
        self.open_blocks = []       # (remaining space, creation order, segments) - will be sorted
        self.next_order = 0

    def add(self, segment):
        segment = np.asarray(segment, dtype=np.uint16)
        if not 0 < len(segment) <= self.seqlen:
            raise ValueError("document segment does not fit in a block")


        i = bisect.bisect_left(self.open_blocks, (len(segment),)) # finding the packed sequence that has smallest space but the current segment fits.
        if i < len(self.open_blocks):
            remaining, order, segments = self.open_blocks.pop(i)
        else:
            remaining, order, segments = self.seqlen, self.next_order, []
            self.next_order += 1

        segments.append(segment)
        remaining -= len(segment)
        if remaining == 0:
            self.write_block(segments)
        else:
            bisect.insort(self.open_blocks, (remaining, order, segments))

        # Do not let incomplete blocks grow without limit. We remove the packed sequence that has the smallest remaining.
        if len(self.open_blocks) > self.buffer_size:
            _, _, segments = self.open_blocks.pop(0)
            self.write_block(segments)

    def finish(self):
        while self.open_blocks:
            _, _, segments = self.open_blocks.pop(0)
            self.write_block(segments)


def make_block(segments, seqlen, pad):
    """Create one fixed-width stored block and its real segment lengths."""
    lengths = np.asarray([len(segment) for segment in segments], dtype=np.uint16)
    used = int(lengths.sum())
    if not segments or used > seqlen:
        raise ValueError("invalid packed block")

    block = np.full(seqlen, pad, dtype=np.uint16)
    position = 0
    for segment in segments:
        block[position:position + len(segment)] = segment
        position += len(segment)
    return block, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="train", choices=["train", "eval_random", "eval_temporal"])
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="maximum real tokens including [CLS]/[SEP]; 0 = no cap")
    ap.add_argument("--source-weights", default="", help="e.g. News=1.0,Gigaword=0.5,Wikipedia=0.3")
    ap.add_argument("--tokenizer", default=MODERNBERT)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--packing-buffer", type=int, default=1000,
                    help="maximum number of incomplete blocks kept for best-fit packing")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dedup", action="store_true", help="exact-dedup by text_hash")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not 2 <= args.seqlen <= np.iinfo(np.uint16).max: #aunabil: do we need this if 2 blocks
        ap.error("--seqlen must be between 2 and 65,535")
    if args.batch < 1 or args.packing_buffer < 1:
        ap.error("--batch and --packing-buffer must be positive")

    # Imported here so the small packing helpers can be tested independently.
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    seen = set() if args.dedup else None
    n_dup = 0
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    cls = tok.cls_token_id
    sep = tok.sep_token_id
    pad = tok.pad_token_id
    if cls is None or sep is None or pad is None:
        raise ValueError("tokenizer must define CLS, SEP, and padding tokens")
    if max(len(tok) - 1, cls, sep, pad) > np.iinfo(np.uint16).max: #aunabil: safety check
        raise ValueError("tokenizer vocabulary does not fit in uint16")
    content_length = args.seqlen - tok.num_special_tokens_to_add(pair=False)
    if content_length < 1:
        raise ValueError("seqlen must leave room for content and special tokens")

    weights = parse_weights(args.source_weights)
    rng = random.Random(args.seed)
    files = sorted(glob.glob(os.path.join(args.corpus, "**", "*.parquet"), recursive=True)) #aunabil: if too many files to process, we need to split it
    rng.shuffle(files)

    token_path = os.path.join(args.out, "tokens.u16")
    length_path = os.path.join(args.out, "segment_lengths.u16")
    offset_path = os.path.join(args.out, "block_segment_offsets.u64")
    token_file = open(token_path, "wb")
    length_file = open(length_path, "wb")
    offset_file = open(offset_path, "wb")
    offset_file.write(np.asarray([0], dtype=np.uint64).tobytes())

    n_blocks = 0
    n_tokens = 0
    n_segments = 0
    n_docs = 0
    n_split_docs = 0
    selected_tokens = 0
    stop = False

    def write_block(segments):
        nonlocal n_blocks, n_tokens, n_segments
        block, lengths = make_block(segments, args.seqlen, pad)
        token_file.write(block.tobytes())
        length_file.write(lengths.tobytes())
        n_blocks += 1
        n_tokens += int(lengths.sum())
        n_segments += len(lengths)
        offset_file.write(np.asarray([n_segments], dtype=np.uint64).tobytes())

    packer = GreedyBestFitPacker(args.seqlen, args.packing_buffer, write_block)

    def pack_batch(text_batch):
        """Tokenize complete documents and return True when the budget is met."""
        nonlocal n_docs, n_split_docs, selected_tokens
        encoded = tok(text_batch, add_special_tokens=False, truncation=False,
                      return_attention_mask=False)["input_ids"]
        for ids in encoded:
            segments = document_segments(ids, cls, sep, content_length)
            if not segments:
                continue
            n_docs += 1
            n_split_docs += len(segments) > 1
            for segment in segments:
                packer.add(segment)
                selected_tokens += len(segment)
            # Keep all pieces of the final document instead of truncating it.
            if args.max_tokens and selected_tokens >= args.max_tokens:
                return True
        return False

    try:
        for f in files: #aunabil: if too many files to process, we need to split it
            if stop:
                break
            src = os.path.basename(os.path.dirname(f))
            keep = 1.0 if weights is None else weights.get(src, 0.0)
            if keep <= 0.0:
                continue
            cols = ["source", "text", "split"] + (["text_hash"] if seen is not None else [])
            t = pq.read_table(f, columns=cols)
            texts = t.column("text").to_pylist() 
            splits = t.column("split").to_pylist()
            hashes = t.column("text_hash").to_pylist() if seen is not None else None
            batch = []
            for i in range(len(texts)):
                if splits[i] != args.split:
                    continue
                if keep < 1.0 and rng.random() > keep:
                    continue
                if seen is not None:
                    if hashes[i] in seen:
                        n_dup += 1
                        continue
                    seen.add(hashes[i])
                batch.append(texts[i])
                if len(batch) >= args.batch:
                    stop = pack_batch(batch)
                    batch = []
                    if stop:
                        break
            if batch and not stop:
                stop = pack_batch(batch)
            print(f"  {src:12s} {os.path.basename(f):32s} blocks={n_blocks:,} "
                  f"tokens={selected_tokens/1e6:.1f}M docs={n_docs:,}", flush=True)

        packer.finish()
    finally:
        token_file.close()
        length_file.close()
        offset_file.close()

    if not n_blocks:
        raise ValueError(f"no documents from split {args.split!r} were packed")

    storage_tokens = n_blocks * args.seqlen
    meta = {
        "format_version": 2,
        "path": os.path.basename(token_path), "dtype": "uint16",
        "segment_lengths_path": os.path.basename(length_path),
        "segment_lengths_dtype": "uint16",
        "block_segment_offsets_path": os.path.basename(offset_path),
        "block_segment_offsets_dtype": "uint64",
        "seqlen": args.seqlen, "n_blocks": n_blocks, "n_tokens": n_tokens,
        "n_storage_tokens": storage_tokens,
        "packing_efficiency": n_tokens / storage_tokens,
        "n_docs": n_docs, "n_segments": n_segments, "n_split_docs": n_split_docs,
        "cls_token_id": cls, "separator_token_id": sep, "pad_token_id": pad,
        "split": args.split, "tokenizer": args.tokenizer,
        "source_weights": weights, "max_tokens": args.max_tokens,
        "dedup": bool(seen is not None), "n_duplicates_skipped": n_dup,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nPACK_DONE blocks={n_blocks:,} real_tokens={n_tokens:,} "
          f"efficiency={meta['packing_efficiency']:.2%} docs={n_docs:,} "
          f"segments={n_segments:,} split_docs={n_split_docs:,} -> {args.out}")


if __name__ == "__main__":
    main()
