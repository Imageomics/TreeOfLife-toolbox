#!/usr/bin/env python3
"""
Cleanup helper for TreeOfLife conversion batches that were staged via symlinks.

The symlink batch contains pointers to the canonical hybrid-format files that
were copied into research storage. Once verification succeeds we remove the
symlink placeholders, the real files they reference, and the legacy parquet
artifacts under the destination tree.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Sequence, Set, Tuple

SymlinkEntry = Tuple[str, str]  # (symlink_path, target_path)

PASS_LINE_RE = re.compile(
    r"^PASS UUID ([0-9a-fA-F-]+) - "
    r"Source rows: (?:\d+|NA), "
    r"Metadata rows: (?:\d+|NA), "
    r"Source images: (?:\d+|NA), "
    r"HDF5 images: (?:\d+|NA), "
    r"Images verified: (?:\d+|NA), "
    r"Invalid images: \d+, "
    r"UUID mismatches: \d+"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cleanup script for symlink-based batches. Removes the symlinked files, "
            "their backing targets, and destination legacy artifacts once verification succeeds."
        )
    )
    parser.add_argument("--source-root", required=True, help="Scratch directory with hybrid outputs")
    parser.add_argument("--dest-root", required=True, help="Destination directory holding final files")
    parser.add_argument("--logs", required=True, help="Logs directory supplied to verification step")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute and record what would be deleted",
    )
    parser.add_argument(
        "--yes-delete",
        action="store_true",
        help="Perform deletion (requires prior dry-run summary)",
    )
    parser.add_argument(
        "--summary-file",
        help="Optional explicit path to *_verification_summary.json inside --logs",
    )
    return parser.parse_args()


def error(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def find_summary(logs_dir: str, summary_file: str | None) -> str:
    if summary_file:
        path = os.path.abspath(summary_file)
        if not os.path.exists(path):
            error(f"Specified summary file not found: {path}")
        return path

    matches = sorted(glob.glob(os.path.join(logs_dir, "*_verification_summary.json")))
    if not matches:
        error("No *_verification_summary.json found in logs directory")
    if len(matches) > 1:
        error("Multiple verification summaries found; specify one via --summary-file")
    return matches[0]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_summary_clean(summary: dict) -> None:
    required_zero_fields = [
        "metadata_failures",
        "image_failures",
        "metadata_errors",
        "image_errors",
        "total_invalid_images",
        "total_uuid_mismatches",
    ]
    total_sets = summary.get("total_sets")
    passes = summary.get("passes")
    if total_sets is None or passes is None:
        error("Summary missing total_sets/passes")
    if int(total_sets) != int(passes):
        error("Not all file sets passed verification")
    for key in required_zero_fields:
        if int(summary.get(key, -1)) != 0:
            error(f"Summary field {key} must be zero to continue")


def ensure_pass_logs(logs_dir: str, uuids: Set[str]) -> None:
    for uuid in sorted(uuids):
        path = os.path.join(logs_dir, f"{uuid}_success.log")
        if not os.path.exists(path):
            error(f"Missing success log for UUID {uuid}")
        found = False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if PASS_LINE_RE.match(line.strip()):
                    found = True
                    break
        if not found:
            error(f"Success log {path} does not contain PASS summary")


def discover_server_paths(root: str, required: Set[str] | None = None) -> Dict[str, str]:
    root = os.path.abspath(root)
    server_paths: Dict[str, str] = {}
    root_name = os.path.basename(root.rstrip(os.sep))

    if root_name.startswith("server="):
        server_paths[root_name] = root
    else:
        try:
            entries = list(os.scandir(root))
        except FileNotFoundError:
            error(f"Root directory not found: {root}")
        for entry in entries:
            if entry.is_dir() and entry.name.startswith("server="):
                if required is None or entry.name in required:
                    server_paths[entry.name] = entry.path

    if required is not None:
        missing = required - set(server_paths.keys())
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            error(f"Missing server directories in {root}: {sample}")
    elif not server_paths:
        error(f"No server= directories found under {root}")

    return server_paths


def scan_symlink_servers(server_paths: Dict[str, str]) -> Dict[str, Dict[str, List[SymlinkEntry]]]:
    data: Dict[str, Dict[str, List[SymlinkEntry]]] = {}
    for server_name, server_path in server_paths.items():
        for dirpath, _, filenames in os.walk(server_path):
            for fname in filenames:
                if not fname.startswith("data_"):
                    continue
                full_path = os.path.join(dirpath, fname)
                target_path = os.path.realpath(full_path)
                if fname.endswith("_metadata.parquet"):
                    uuid = fname[5:-len("_metadata.parquet")]
                    data.setdefault(uuid, {}).setdefault("metadata", []).append((full_path, target_path))
                elif fname.endswith("_images.h5"):
                    uuid = fname[5:-len("_images.h5")]
                    data.setdefault(uuid, {}).setdefault("images", []).append((full_path, target_path))
    return data


def scan_dest_servers(server_paths: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    data: Dict[str, Dict[str, List[str]]] = {}
    for server_name, server_path in server_paths.items():
        for dirpath, _, filenames in os.walk(server_path):
            for fname in filenames:
                if not fname.startswith("data_"):
                    continue
                full_path = os.path.join(dirpath, fname)
                if fname.endswith("_metadata.parquet"):
                    uuid = fname[5:-len("_metadata.parquet")]
                    data.setdefault(uuid, {}).setdefault("metadata", []).append(full_path)
                elif fname.endswith("_images.h5"):
                    uuid = fname[5:-len("_images.h5")]
                    data.setdefault(uuid, {}).setdefault("images", []).append(full_path)
                elif fname.endswith(".parquet"):
                    uuid = fname[5:-len(".parquet")]
                    data.setdefault(uuid, {}).setdefault("legacy", []).append(full_path)
    return data


def ensure_single_symlink(label: str, files: List[SymlinkEntry], uuid: str) -> SymlinkEntry:
    if not files:
        error(f"Missing {label} for {uuid}")
    if len(files) > 1:
        error(f"Multiple {label} files found for {uuid}")
    return files[0]


def ensure_single(label: str, files: List[str], uuid: str) -> str:
    if not files:
        error(f"Missing {label} for {uuid}")
    if len(files) > 1:
        error(f"Multiple {label} files found for {uuid}")
    return files[0]


def build_deletion_plan(
    target_uuids: Set[str],
    source_map: Dict[str, Dict[str, List[SymlinkEntry]]],
    dest_map: Dict[str, Dict[str, List[str]]],
) -> Tuple[List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]]]:
    source_entries: List[Tuple[str, str, str, str]] = []
    dest_entries: List[Tuple[str, str, str, str]] = []
    for uuid in sorted(target_uuids):
        s_meta_symlink, s_meta_target = ensure_single_symlink(
            "source metadata", source_map.get(uuid, {}).get("metadata", []), uuid
        )
        s_img_symlink, s_img_target = ensure_single_symlink(
            "source images", source_map.get(uuid, {}).get("images", []), uuid
        )
        d_old = ensure_single("dest legacy parquet", dest_map.get(uuid, {}).get("legacy", []), uuid)
        source_entries.append((uuid, "source_symlink", "metadata", s_meta_symlink))
        source_entries.append((uuid, "source_target", "metadata", s_meta_target))
        source_entries.append((uuid, "source_symlink", "images", s_img_symlink))
        source_entries.append((uuid, "source_target", "images", s_img_target))
        dest_entries.append((uuid, "dest", "legacy_parquet", d_old))
    return source_entries, dest_entries


def ensure_uuid_parity(
    expected_uuids: Set[str],
    source_map: Dict[str, Dict[str, List[SymlinkEntry]]],
    dest_map: Dict[str, Dict[str, List[str]]],
) -> None:
    source_set = set(source_map.keys())
    dest_set = set(dest_map.keys())
    if source_set != expected_uuids:
        error("Source UUID set does not match expected batch UUIDs")
    if dest_set != expected_uuids:
        error("Dest UUID set does not match expected batch UUIDs")


def ensure_dest_metadata_completeness(
    target_uuids: Set[str],
    source_map: Dict[str, Dict[str, List[SymlinkEntry]]],
    dest_map: Dict[str, Dict[str, List[str]]],
) -> None:
    for uuid in sorted(target_uuids):
        s_meta_symlink, _ = ensure_single_symlink(
            "source metadata", source_map.get(uuid, {}).get("metadata", []), uuid
        )
        s_img_symlink, _ = ensure_single_symlink(
            "source images", source_map.get(uuid, {}).get("images", []), uuid
        )
        d_meta = ensure_single("dest metadata", dest_map.get(uuid, {}).get("metadata", []), uuid)
        d_img = ensure_single("dest images", dest_map.get(uuid, {}).get("images", []), uuid)
        if os.path.basename(s_meta_symlink) != os.path.basename(d_meta):
            error(
                "Dest metadata filename mismatch for "
                f"{uuid}: {os.path.basename(d_meta)} vs {os.path.basename(s_meta_symlink)}"
            )
        if os.path.basename(s_img_symlink) != os.path.basename(d_img):
            error(
                "Dest image filename mismatch for "
                f"{uuid}: {os.path.basename(d_img)} vs {os.path.basename(s_img_symlink)}"
            )


def manifest(entries: Sequence[Tuple[str, str, str, str]]) -> Tuple[List[str], str]:
    formatted = [f"{uuid}|{loc}|{kind}|{path}" for uuid, loc, kind, path in entries]
    h = hashlib.sha256()
    for row in formatted:
        h.update(row.encode("utf-8"))
        h.update(b"\n")
    return formatted, h.hexdigest()


def write_dry_run(
    args: argparse.Namespace,
    summary_path: str,
    source_entries: List[Tuple[str, str, str, str]],
    dest_entries: List[Tuple[str, str, str, str]],
) -> None:
    csv_path = summary_path.replace(
        "_verification_summary.json", "_cleanup_dry_run_files.csv"
    )
    json_path = summary_path.replace(
        "_verification_summary.json", "_cleanup_dry_run_summary.json"
    )
    total_entries = source_entries + dest_entries
    _, manifest_hash = manifest(total_entries)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["uuid", "location", "file_type", "path"])
        for uuid, loc, kind, path in total_entries:
            writer.writerow([uuid, loc, kind, path])

    uuid_count = len({uuid for uuid, _, _, _ in total_entries})

    summary_payload = {
        "mode": "dry-run",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "args": {
            "source_root": os.path.abspath(args.source_root),
            "dest_root": os.path.abspath(args.dest_root),
            "logs": os.path.abspath(args.logs),
        },
        "verification_summary": os.path.abspath(summary_path),
        "csv_listing": os.path.abspath(csv_path),
        "uuid_count": uuid_count,
        "source_files": len(source_entries),
        "dest_files": len(dest_entries),
        "total_files": len(total_entries),
        "file_manifest_sha256": manifest_hash,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Dry-run summary: {json_path}")
    print(f"Dry-run CSV: {csv_path}")


def load_dry_run(summary_path: str) -> dict:
    dry_run = summary_path.replace(
        "_verification_summary.json", "_cleanup_dry_run_summary.json"
    )
    if not os.path.exists(dry_run):
        error("Dry-run summary missing; run with --dry-run first")
    return load_json(dry_run)


def read_csv_entries(csv_path: str) -> List[Tuple[str, str, str, str]]:
    entries: List[Tuple[str, str, str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            entries.append((row["uuid"], row["location"], row["file_type"], row["path"]))
    return entries


def perform_deletions(entries: Sequence[Tuple[str, str, str, str]]) -> None:
    for uuid, loc, kind, path in entries:
        exists = os.path.lexists(path) if "symlink" in loc else os.path.exists(path)
        if not exists:
            error(f"Expected file missing before deletion: {path}")
    deleted: Set[str] = set()
    for uuid, loc, kind, path in entries:
        if path in deleted:
            continue
        os.remove(path)
        deleted.add(path)
        print(f"Deleted [{loc}][{kind}] {path}")


def main() -> int:
    args = parse_args()

    if args.dry_run == args.yes_delete:
        error("Specify exactly one of --dry-run or --yes-delete")

    source_root = os.path.abspath(args.source_root)
    dest_root = os.path.abspath(args.dest_root)
    logs_dir = os.path.abspath(args.logs)

    for path, label in (
        (source_root, "source-root"),
        (dest_root, "dest-root"),
        (logs_dir, "logs"),
    ):
        if not os.path.isdir(path):
            error(f"{label} directory not found: {path}")

    summary_path = find_summary(logs_dir, args.summary_file)
    summary = load_json(summary_path)
    ensure_summary_clean(summary)

    per_uuid = summary.get("per_uuid")
    if not per_uuid:
        error("Summary missing per_uuid data")
    summary_uuids = set(per_uuid.keys())
    if not summary_uuids:
        error("Summary UUID set empty")

    ensure_pass_logs(logs_dir, summary_uuids)

    source_servers = discover_server_paths(source_root)
    dest_servers = discover_server_paths(dest_root, required=set(source_servers.keys()))

    source_map = scan_symlink_servers(source_servers)
    if not source_map:
        error("No data_* files discovered under source-root")
    batch_uuids = set(source_map.keys())
    missing_from_summary = batch_uuids - summary_uuids
    if missing_from_summary:
        sample = ", ".join(sorted(missing_from_summary)[:5])
        error(f"Batch UUIDs missing from verification summary: {sample}")

    dest_map_all = scan_dest_servers(dest_servers)
    missing_dest = [uuid for uuid in batch_uuids if uuid not in dest_map_all]
    if missing_dest:
        sample = ", ".join(sorted(missing_dest)[:5])
        error(f"Destination missing UUIDs for batch: {sample}")
    dest_map = {uuid: dest_map_all[uuid] for uuid in batch_uuids}

    ensure_uuid_parity(batch_uuids, source_map, dest_map)
    ensure_dest_metadata_completeness(batch_uuids, source_map, dest_map)

    source_entries, dest_entries = build_deletion_plan(batch_uuids, source_map, dest_map)
    total_entries = source_entries + dest_entries

    if args.dry_run:
        write_dry_run(args, summary_path, source_entries, dest_entries)
        return 0

    dry_run_data = load_dry_run(summary_path)
    recorded_args = dry_run_data.get("args", {})
    current_args = {
        "source_root": source_root,
        "dest_root": dest_root,
        "logs": logs_dir,
    }
    if recorded_args != current_args:
        error("Current arguments do not match dry-run summary")

    csv_path = dry_run_data.get("csv_listing")
    if not csv_path or not os.path.exists(csv_path):
        error("Dry-run CSV listing missing; rerun dry-run")

    recorded_entries = read_csv_entries(csv_path)
    _, recorded_hash = manifest(recorded_entries)
    _, current_hash = manifest(total_entries)
    if recorded_hash != dry_run_data.get("file_manifest_sha256"):
        error("Dry-run manifest hash mismatch; rerun dry-run")
    if recorded_hash != current_hash:
        error("Filesystem layout changed since dry-run; rerun dry-run")

    print(f"Deleting {len(recorded_entries)} files...")
    perform_deletions(recorded_entries)
    print("Cleanup completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
