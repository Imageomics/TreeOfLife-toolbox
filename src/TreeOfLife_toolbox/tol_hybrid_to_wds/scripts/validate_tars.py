#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import time
import tarfile
from collections import defaultdict

# The 10 required sidecar suffixes
REQUIRED_SIDECARS = [
    'com.txt',
    'common_name.txt',
    'sci.txt',
    'sci_com.txt',
    'scientific_name.txt',
    'taxon.txt',
    'taxonTag.txt',
    'taxonTag_com.txt',
    'taxon_com.txt',
    'taxonomic_name.txt',
]

def validate_tar(tar_path: str) -> tuple[str, dict]:
    """
    Validate that each JPG has all 10 required text sidecars AND they contain text.
    Returns (tar_name, validation_result)
    """
    tar_name = os.path.basename(tar_path)
    
    # Track all files by UUID
    files_by_uuid = defaultdict(dict)  # uuid -> {extension: has_content}
    jpg_uuids = set()
    
    try:
        with tarfile.open(tar_path, 'r|') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                
                name = member.name
                basename = os.path.basename(name)
                
                if '.' not in basename:
                    continue
                
                # Split on first dot to get UUID
                parts = basename.split('.', 1)
                if len(parts) != 2:
                    continue
                
                uuid = parts[0]
                extension = parts[1]
                
                # Track JPGs
                if extension.lower() == 'jpg':
                    jpg_uuids.add(uuid)
                    files_by_uuid[uuid][extension] = True  # JPGs don't need content check
                
                # For text files, check if they have content
                elif extension in REQUIRED_SIDECARS:
                    has_content = False
                    try:
                        f = tar.extractfile(member)
                        if f:
                            content = f.read()
                            # Check if file has any non-whitespace content
                            has_content = len(content.strip()) > 0
                            f.close()
                    except Exception:
                        has_content = False
                    
                    files_by_uuid[uuid][extension] = has_content
        
        # Now validate: check each JPG has all sidecars with content
        issues = []
        for uuid in sorted(jpg_uuids):
            extensions = files_by_uuid[uuid]
            missing = []
            empty = []
            
            for required in REQUIRED_SIDECARS:
                if required not in extensions:
                    missing.append(required)
                elif not extensions[required]:
                    empty.append(required)
            
            if missing or empty:
                issue = {'uuid': uuid}
                if missing:
                    issue['missing'] = missing
                if empty:
                    issue['empty'] = empty
                issues.append(issue)
        
        return (tar_name, {
            'valid': len(issues) == 0,
            'jpg_count': len(jpg_uuids),
            'issues': issues
        })
    
    except Exception as e:
        return (tar_name, {
            'valid': False,
            'error': str(e),
            'jpg_count': 0,
            'issues': []
        })

def process_chunk(tar_paths: list[str]) -> list[tuple[str, dict]]:
    """Process multiple tars in one worker."""
    return [validate_tar(p) for p in tar_paths]

def chunks(lst, n):
    """Yield successive n-sized chunks."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def validate_all_tars(source_dir: str, workers: int, chunk_size: int = 1) -> None:
    src = Path(source_dir)
    tar_files = sorted([str(p) for p in src.glob("*.tar")])
    n = len(tar_files)
    
    if n == 0:
        raise SystemExit(f"No .tar files in {src}")
    
    print(f"Validating {n} shards with {workers} workers...", flush=True)
    print("Checking: file existence + non-empty content", flush=True)
    
    results = {}
    all_issues = []
    total_jpgs = 0
    valid_tars = 0
    total_missing = 0
    total_empty = 0
    
    t0 = time()
    processed = 0
    
    # Split work into chunks
    tar_chunks = list(chunks(tar_files, chunk_size))
    
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for result_chunk in ex.map(process_chunk, tar_chunks):
            for tar_name, validation in result_chunk:
                results[tar_name] = validation
                
                jpg_count = validation.get('jpg_count', 0)
                total_jpgs += jpg_count
                
                if validation['valid']:
                    valid_tars += 1
                else:
                    # Record issues for this tar
                    for issue in validation.get('issues', []):
                        issue_record = {
                            'tar': tar_name,
                            'uuid': issue['uuid']
                        }
                        if 'missing' in issue:
                            issue_record['missing'] = issue['missing']
                            total_missing += len(issue['missing'])
                        if 'empty' in issue:
                            issue_record['empty'] = issue['empty']
                            total_empty += len(issue['empty'])
                        all_issues.append(issue_record)
                    
                    # Also note if there was an error
                    if 'error' in validation:
                        all_issues.append({
                            'tar': tar_name,
                            'error': validation['error']
                        })
                
                processed += 1
                if processed % 500 == 0 or processed == n:
                    elapsed = time() - t0
                    rate = processed / elapsed
                    eta = (n - processed) / rate if rate > 0 else 0
                    print(f"[{processed}/{n}] {rate:.1f}/s | Valid: {valid_tars}/{processed} | Issues: {len(all_issues)} (missing: {total_missing}, empty: {total_empty}) | ETA: {eta:.0f}s", flush=True)
    
    # Write detailed results
    out_path = src / "tar_validation_results.json"
    summary = {
        'summary': {
            'total_tars': n,
            'valid_tars': valid_tars,
            'invalid_tars': n - valid_tars,
            'total_jpgs': total_jpgs,
            'total_issues': len(all_issues),
            'total_missing_files': total_missing,
            'total_empty_files': total_empty
        },
        'per_tar': results
    }
    
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Write issue log (more human-readable)
    if all_issues:
        issues_path = src / "tar_validation_issues.log"
        with open(issues_path, "w") as f:
            f.write(f"Validation Issues - {len(all_issues)} total\n")
            f.write(f"Missing files: {total_missing}, Empty files: {total_empty}\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by tar file
            by_tar = defaultdict(list)
            for issue in all_issues:
                by_tar[issue['tar']].append(issue)
            
            for tar_name in sorted(by_tar.keys()):
                issues = by_tar[tar_name]
                f.write(f"TAR: {tar_name} ({len(issues)} issues)\n")
                f.write("-" * 80 + "\n")
                
                for issue in issues:
                    if 'error' in issue:
                        f.write(f"  ERROR: {issue['error']}\n")
                    else:
                        f.write(f"  UUID: {issue['uuid']}\n")
                        if 'missing' in issue:
                            f.write(f"    Missing: {', '.join(issue['missing'])}\n")
                        if 'empty' in issue:
                            f.write(f"    Empty: {', '.join(issue['empty'])}\n")
                
                f.write("\n")
        
        print(f"\n⚠️  Issues found: {len(all_issues)}")
        print(f"    Missing files: {total_missing}")
        print(f"    Empty files: {total_empty}")
        print(f"    Wrote details to: {issues_path}")
    
    elapsed = time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s ({n/elapsed:.1f} shards/s)")
    print(f"✓ Valid tars: {valid_tars}/{n} ({100*valid_tars/n:.1f}%)")
    print(f"✓ Total JPGs: {total_jpgs:,}")
    print(f"✓ Wrote summary to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate JPG sidecars in tar shards (checks content)")
    ap.add_argument("--source_dir", required=True, help="Directory containing .tar files")
    ap.add_argument("--workers", type=int, default=96, help="Number of parallel workers")
    ap.add_argument("--chunk-size", type=int, default=1, help="Tars per worker batch")
    args = ap.parse_args()
    
    validate_all_tars(args.source_dir, args.workers, args.chunk_size)

