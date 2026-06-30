#!/usr/bin/env python3
"""
wds_taxoncom_audit.py

One-node audit / training-pipeline analog to pinpoint where samples are missing `taxon_com.txt`.

Logs:
  - wds_audit_logs/missing_taxon_com.workerXYZ.jsonl  (samples missing taxon_com.txt after grouping)
  - wds_audit_logs/bad_records.workerXYZ.jsonl        (non-empty dicts missing fname/data)
  - wds_audit_logs/stats.workerXYZ.json               (summary per worker)

Usage:
  python wds_taxoncom_audit.py \
    --shards "/path/to/shards/shard-{00000..99999}.tar" \
    --workers 96 \
    --out-dir wds_audit_logs
"""

import argparse
import json
import logging
import os
from datetime import datetime

import webdataset as wds
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample


# ----------------------------
# Logging / utilities
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def now_ts():
    return datetime.now().isoformat(timespec="seconds")


def worker_id():
    """
    Best-effort worker id detection.

    Note: torch dataloader workers don't reliably export a standard env var.
    We'll use these if present, else -1.
    """
    for k in ("WDS_WORKER", "WORKER", "LOCAL_RANK", "RANK"):
        v = os.environ.get(k)
        if v is not None and str(v).lstrip("-").isdigit():
            return int(v)
    return -1


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ----------------------------
# WDS "nothrow" tar parsing
# ----------------------------

def log_and_continue(exn):
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


# These globals get set in main() so that nested functions can log without threading args everywhere.
OUT_DIR = "wds_audit_logs"


def log_bad_record(filesample, reason="missing_fname_or_data"):
    wid = worker_id()
    path = os.path.join(OUT_DIR, f"bad_records.worker{wid:03d}.jsonl")
    rec = {
        "ts": now_ts(),
        "reason": reason,
        "url": filesample.get("__url__", filesample.get("url")),
        "keys": sorted(list(filesample.keys())),
    }
    append_jsonl(path, rec)


def log_missing_taxon_com(sample, reason="missing taxon_com.txt (pre-rename)"):
    wid = worker_id()
    path = os.path.join(OUT_DIR, f"missing_taxon_com.worker{wid:03d}.jsonl")
    rec = {
        "ts": now_ts(),
        "reason": reason,
        "url": sample.get("__url__", sample.get("url")),
        "key": sample.get("__key__"),
        "present_suffixes": sorted([k for k in sample.keys() if not k.startswith("__")]),
    }
    append_jsonl(path, rec)


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """
    Robust grouping that tolerates:
      - empty dicts (often sentinel-ish): skip silently
      - malformed dicts missing fname/data: log (if non-empty) and skip

    This is meant to be a faithful-ish version of the modified training pipeline,
    while remaining diagnostic.
    """
    current_sample = None

    for filesample in data:
        if not isinstance(filesample, dict):
            logging.warning(f"Skipping non-dict filesample: {type(filesample)}")
            continue

        # IMPORTANT: empty dicts can appear due to iterator plumbing; skip quietly.
        if not filesample:
            continue

        # Defensive: sometimes expander yields dicts without fname/data
        if "fname" not in filesample or "data" not in filesample:
            # Log only non-empty dicts (otherwise it's useless noise).
            log_bad_record(filesample, reason="malformed_tar_record_missing_fname_or_data")
            continue

        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()

        # Same collision handling rationale as training data.py
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample.get("__url__"))

        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value

    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    return group_by_keys_nothrow(files, handler=handler)


# ----------------------------
# Filters for audit
# ----------------------------

def has_image(sample):
    return ("png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample)


def has_any_txt(sample):
    # training's filter_no_caption_or_no_image checks "any('txt' in key for key in sample)"
    return any("txt" in k for k in sample.keys())


def require_taxon_com_txt(sample):
    """
    Audit hook: record and drop samples that lack taxon_com.txt at the grouped-sample stage.

    This is the strongest check for grouping/boundary/corruption-induced misses.
    """
    if "taxon_com.txt" not in sample:
        log_missing_taxon_com(sample, reason="missing taxon_com.txt (pre-rename)")
        return False
    return True


# ----------------------------
# Main audit
# ----------------------------

def audit(shards_pattern: str, workers: int, log_every: int, limit_kept: int):
    shards = list(wds.shardlists.expand_urls(shards_pattern))
    logging.info(f"Expanded to {len(shards)} shard URLs")

    # Build a training-ish pipeline (single node; split_by_node not needed here)
    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
        wds.select(lambda s: has_any_txt(s) and has_image(s)),
        wds.select(lambda s: require_taxon_com_txt(s)),
        # We intentionally avoid decode/tokenizer here: this audit is about tar member presence.
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        persistent_workers=True,
    )

    kept = 0
    for sample in loader:
        kept += 1
        if log_every and kept % log_every == 0:
            logging.info(f"Kept {kept} samples (after filters)")

        if limit_kept and kept >= limit_kept:
            break

    logging.info(f"Done. Kept {kept} samples (after filters). Logs in {OUT_DIR}/")


def main():
    global OUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help='Braceexpand pattern like ".../shard-{00000..99999}.tar"')
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out-dir", default="wds_audit_logs")
    ap.add_argument("--log-every", type=int, default=200000)
    ap.add_argument("--limit", type=int, default=0, help="Stop after this many kept samples (0 = no limit)")
    args = ap.parse_args()

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    audit(args.shards, args.workers, args.log_every, args.limit)


if __name__ == "__main__":
    main()
