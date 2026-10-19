"""Write TreeOfLife samples into a Lance dataset.

Worker processes turn each tar shard into one uncommitted Lance fragment (write_fragment). The main process then
commits all fragments at once, in shard order (commit), and writes the manifest the bioclip-2 loader reads
(write_manifest).
"""
import itertools
import json
import os
from typing import Iterator, NamedTuple

import lance
import pyarrow as pa
from lance.fragment import FragmentMetadata, LanceFragment

from tar_reader import read_samples

# The ten caption files, in column order: <key>.sci.txt becomes column "sci", and so on
TEXT_COLUMNS = ["scientific_name", "common_name", "taxonomic_name", "sci", "com", "taxon",
                "sci_com", "taxon_com", "taxonTag", "taxonTag_com"]
MEMBERS = {"jpg", *(f"{column}.txt" for column in TEXT_COLUMNS)}  # every sample must have exactly these
SCHEMA = pa.schema([pa.field(name, dtype, nullable=False) for name, dtype in [
    ("uuid", pa.string()),          # the sample key
    ("jpg", pa.large_binary()),     # the image bytes, unchanged
    *[(column, pa.string()) for column in TEXT_COLUMNS],
    ("source_shard", pa.string()),  # the tar file the sample came from
    ("source_index", pa.uint32()),  # the sample's position in that tar
]])
BATCH_ROWS = 512  # rows a worker holds in memory at a time
# Pin the file format: a newer pylance would otherwise write files the training environment cannot read
DATA_STORAGE_VERSION = "2.1"


class Shard(NamedTuple):
    """A worker's result for one tar. Its fragment travels to the main process as JSON; None for an empty tar."""
    name: str
    rows: int
    fragment: dict | None


def write_fragment(tar_path: str, output: str) -> Shard:
    """Write one tar shard as one uncommitted fragment of the dataset at `output`. Runs in a worker process."""
    name = os.path.basename(tar_path)
    batches = record_batches(tar_path)
    first = next(batches, None)  # peek: an empty tar has no batches and gets no fragment
    if first is None:
        return Shard(name, 0, None)
    reader = pa.RecordBatchReader.from_batches(SCHEMA, itertools.chain([first], batches))
    fragment = LanceFragment.create(output, reader, schema=SCHEMA, data_storage_version=DATA_STORAGE_VERSION)
    return Shard(name, fragment.num_rows, fragment.to_json())


def record_batches(tar_path: str) -> Iterator[pa.RecordBatch]:
    """Yield the shard's samples as record batches of up to BATCH_ROWS rows, in tar order."""
    shard = os.path.basename(tar_path)
    rows = []
    for index, (key, sample) in enumerate(read_samples(tar_path)):
        rows.append(to_row(shard, index, key, sample))
        if len(rows) == BATCH_ROWS:
            yield to_batch(rows)
            rows = []
    if rows:
        yield to_batch(rows)


def to_row(shard: str, index: int, key: str, sample: dict[str, bytes]) -> list:
    """Return one sample as a row in SCHEMA order, or raise ValueError if it does not have exactly MEMBERS."""
    if sample.keys() != MEMBERS:
        missing = ", ".join(sorted(MEMBERS - sample.keys())) or "none"
        unexpected = ", ".join(sorted(sample.keys() - MEMBERS)) or "none"
        raise ValueError(f"{shard}: sample {key} does not have the 11 TreeOfLife members "
                         f"(missing: {missing}; unexpected: {unexpected})")
    captions = [sample[f"{column}.txt"].decode("utf-8") for column in TEXT_COLUMNS]
    return [key, sample["jpg"], *captions, shard, index]


def to_batch(rows: list[list]) -> pa.RecordBatch:
    """Turn a list of rows, each in SCHEMA order, into one record batch."""
    columns = [pa.array(values, field.type) for values, field in zip(zip(*rows), SCHEMA)]
    return pa.RecordBatch.from_arrays(columns, schema=SCHEMA)


def commit(output: str, shards: list[Shard]) -> None:
    """Create the dataset at `output` from the shards' fragments, in the given order."""
    # pylance 11's to_json() returns a dict, while from_json() takes a string
    fragments = [FragmentMetadata.from_json(json.dumps(shard.fragment)) for shard in shards]
    lance.LanceDataset.commit(output, lance.LanceOperation.Overwrite(SCHEMA, fragments))


def write_manifest(output: str, split_dir: str, shards: list[Shard]) -> str:
    """Write <output>.manifest.json and return its path.

    It holds the source directory, the total row count, and each shard's name, rows and offset (the dataset index
    of its first row).
    """
    entries, offset = [], 0
    for shard in shards:
        entries.append({"name": shard.name, "rows": shard.rows, "offset": offset})
        offset += shard.rows
    path = output + ".manifest.json"
    with open(path, "w") as f:
        json.dump({"source": os.path.abspath(split_dir), "num_rows": offset, "shards": entries}, f, indent=1)
    return path
