import itertools
import json
import os
from typing import Iterator, NamedTuple

import lance
import pyarrow as pa
from lance.fragment import FragmentMetadata, LanceFragment

from tar_reader import read_samples

TEXT_COLUMNS = ["scientific_name", "common_name", "taxonomic_name", "sci", "com", "taxon",
                "sci_com", "taxon_com", "taxonTag", "taxonTag_com"]
MEMBERS = {"jpg", *(f"{column}.txt" for column in TEXT_COLUMNS)}
SCHEMA = pa.schema([pa.field(name, dtype, nullable=False) for name, dtype in [
    ("uuid", pa.string()),
    ("jpg", pa.large_binary()),
    *[(column, pa.string()) for column in TEXT_COLUMNS],
    ("source_shard", pa.string()),
    ("source_index", pa.uint32()),
]])
BATCH_ROWS = 512
DATA_STORAGE_VERSION = "2.1"  # newer pylance writes files the training environment cannot read


class Shard(NamedTuple):
    name: str
    rows: int
    fragment: dict | None


def write_fragment(tar_path: str, output: str) -> Shard:
    name = os.path.basename(tar_path)
    batches = record_batches(tar_path)
    first = next(batches, None)
    if first is None:
        return Shard(name, 0, None)
    reader = pa.RecordBatchReader.from_batches(SCHEMA, itertools.chain([first], batches))
    fragment = LanceFragment.create(output, reader, schema=SCHEMA,
                                    data_storage_version=DATA_STORAGE_VERSION)
    return Shard(name, fragment.num_rows, fragment.to_json())


def record_batches(tar_path: str) -> Iterator[pa.RecordBatch]:
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
    if sample.keys() != MEMBERS:
        missing = ", ".join(sorted(MEMBERS - sample.keys())) or "none"
        unexpected = ", ".join(sorted(sample.keys() - MEMBERS)) or "none"
        raise ValueError(f"{shard}: sample {key} does not have the 11 TreeOfLife members "
                         f"(missing: {missing}; unexpected: {unexpected})")
    captions = [sample[f"{column}.txt"].decode("utf-8") for column in TEXT_COLUMNS]
    return [key, sample["jpg"], *captions, shard, index]


def to_batch(rows: list[list]) -> pa.RecordBatch:
    columns = [pa.array(values, field.type) for values, field in zip(zip(*rows), SCHEMA)]
    return pa.RecordBatch.from_arrays(columns, schema=SCHEMA)


def commit(output: str, shards: list[Shard]) -> None:
    # pylance 11's to_json() returns a dict, while from_json() takes a string
    fragments = [FragmentMetadata.from_json(json.dumps(shard.fragment)) for shard in shards]
    lance.LanceDataset.commit(output, lance.LanceOperation.Overwrite(SCHEMA, fragments))


def write_manifest(output: str, split_dir: str, shards: list[Shard]) -> str:
    entries, offset = [], 0
    for shard in shards:
        entries.append({"name": shard.name, "rows": shard.rows, "offset": offset})
        offset += shard.rows
    path = output + ".manifest.json"
    with open(path, "w") as f:
        manifest = {"source": os.path.abspath(split_dir), "num_rows": offset, "shards": entries}
        json.dump(manifest, f, indent=1)
    return path
