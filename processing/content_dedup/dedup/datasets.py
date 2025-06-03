import bisect
import concurrent.futures
import csv
import glob
import logging
import os
from typing import Annotated, TypeGuard

import beartype
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import wids
from jaxtyping import UInt8

from . import helpers

logger = logging.getLogger(__name__)


@beartype.beartype
class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        raise NotImplementedError()


# Dataset registry to store dataset classes
_DATASET_REGISTRY: dict[str, type[Dataset]] = {}


@beartype.beartype
def register_dataset(key: str):
    """Decorator to register a dataset class."""

    def decorator(cls: type[Dataset]) -> type[Dataset]:
        if key in _DATASET_REGISTRY:
            raise ValueError(f"Dataset with key '{key}' already registered")
        _DATASET_REGISTRY[key] = cls
        return cls

    return decorator


@beartype.beartype
def is_valid_dataset_key(key: str) -> TypeGuard[str]:
    """Runtime validation for dataset keys."""
    return key in _DATASET_REGISTRY


# Define a validated dataset key type
DatasetKey = Annotated[str, is_valid_dataset_key]


@beartype.beartype
def list_datasets() -> list[str]:
    """Return a list of all registered dataset keys."""
    return list(_DATASET_REGISTRY.keys())


@beartype.beartype
class ImageFolder(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        self.root = root
        self.split = split
        self.name_to_i = {}
        self.samples = []

        for cls_name in sorted(os.listdir(os.path.join(root, split))):
            self.name_to_i[cls_name] = len(self.name_to_i)

        for path, dir_names, file_names in os.walk(os.path.join(root, split)):
            if not file_names:
                continue

            for file_name in file_names:
                cls_name = os.path.basename(path)
                if cls_name not in self.name_to_i:
                    raise ValueError(f"Class {cls_name} not in self.name_to_i.")
                self.samples.append((os.path.join(path, file_name), cls_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        fpath, cls_name = self.samples[i]
        img = cv2.imread(fpath)
        # Change from BGR to RGB.
        img = img[..., ::-1]
        cls_i = self.name_to_i[cls_name]
        return img, cls_i


@beartype.beartype
@register_dataset("inat21")
class Inat21(ImageFolder):
    def __repr__(self) -> str:
        return f"inat21/{self.split}"


@beartype.beartype
@register_dataset("herbarium19")
class Herbarium19(ImageFolder):
    def __repr__(self) -> str:
        return f"herbarium/{self.split}"


@beartype.beartype
@register_dataset("plantdoc")
class PlantDoc(ImageFolder):
    def __repr__(self) -> str:
        return f"plantdoc/{self.split}"


@beartype.beartype
@register_dataset("fungi")
class Fungi(ImageFolder):
    def __repr__(self) -> str:
        return f"fungi/{self.split}"


@beartype.beartype
@register_dataset("insects")
class Insects(ImageFolder):
    def __repr__(self) -> str:
        return f"insects/{self.split}"


@beartype.beartype
@register_dataset("insects2")
class Insects2(ImageFolder):
    def __repr__(self) -> str:
        return f"insects2/{self.split}"


@beartype.beartype
@register_dataset("plantnet")
class PlantNet(ImageFolder):
    def __repr__(self) -> str:
        return f"plantnet/{self.split}"


@beartype.beartype
@register_dataset("nabirds")
class NaBirds(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        self.root = root
        self.split = split
        self.samples = []

        id2path = {}
        with open(os.path.join(root, "images.txt")) as fd:
            for line in fd:
                img_id, path = line.split()
                id2path[img_id] = os.path.join(root, path)

        id2cls = {}
        with open(os.path.join(root, "image_class_labels.txt")) as fd:
            for line in fd:
                img_id, cls = line.split()
                id2cls[img_id] = int(cls)

        with open(os.path.join(root, "train_test_split.txt")) as fd:
            for line in fd:
                img_id, is_train = line.split()
                if split == "test" and is_train == "1":
                    continue

                if split == "train" and is_train == "0":
                    continue

                self.samples.append((
                    os.path.join(root, "images", id2path[img_id]),
                    id2cls[img_id],
                ))
                print(self.samples[-1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        fpath, cls_name = self.samples[i]
        img = cv2.imread(fpath)
        # Change from BGR to RGB.
        img = img[..., ::-1]
        cls_i = self.name_to_i[cls_name]
        return img, cls_i

    def __repr__(self) -> str:
        return f"nabirds/{self.split}"


@beartype.beartype
@register_dataset("awa2")
class Awa2(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        self.root = root
        self.split = split
        self.name_to_i = {}
        self.samples = []

        with open(os.path.join(root, f"{split}classes.txt")) as fd:
            split_classes = {line.strip() for line in fd}

        for cls_name in sorted(os.listdir(os.path.join(root, "JPEGImages"))):
            if cls_name not in split_classes:
                continue
            self.name_to_i[cls_name] = len(self.name_to_i)

        for path, dir_names, file_names in os.walk(os.path.join(root, "JPEGImages")):
            if not file_names:
                continue

            for file_name in file_names:
                cls_name = os.path.basename(path)
                if cls_name not in self.name_to_i:
                    continue
                self.samples.append((os.path.join(path, file_name), cls_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"awa2/{self.split}"

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        fpath, cls_name = self.samples[i]
        img = cv2.imread(fpath)
        # Change from BGR to RGB.
        img = img[..., ::-1]
        cls_i = self.name_to_i[cls_name]
        return img, cls_i


@beartype.beartype
@register_dataset("iwildcam")
class IWildCam(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        import wilds

        self.root = root
        self.split = split

        dataset = wilds.get_dataset(dataset="iwildcam", download=False, root_dir=root)
        self.dataset = dataset.get_subset(split, transform=None)

    def __repr__(self) -> str:
        return f"iwildcam/{self.split}"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        img, label, _ = self.dataset[i]
        img = np.array(img)

        return img, label.item()


@beartype.beartype
@register_dataset("tol-10m")
class TreeOfLife(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        self.root = root
        self.split = split
        self.logger = logging.getLogger(repr(self))

        # Find statistics.csv in the parent directory
        stats_path = os.path.join(root, "statistics.csv")
        # Create class lookup from statistics.csv
        self.cls_to_idx, self.idx_to_cls = _make_tol_class_lookup(stats_path)

        # Find all TAR files in the given directory
        shard_pattern = os.path.join(root, split, "*.tar")
        shards = glob.glob(shard_pattern)

        if not shards:
            raise ValueError(f"No shards found at {shard_pattern}")

        # Create a list of shard descriptors for ShardListDataset
        shards_info = []
        for shard in shards:
            try:
                # Get number of samples in the shard
                n_samples = wids.wids.compute_num_samples(shard)
                shards_info.append({"url": shard, "nsamples": n_samples})
            except Exception as err:
                logger.warning("Error processing shard '%s': %s", shard, err)

        if not shards_info:
            raise ValueError(f"No valid shards found in {shard_pattern}")

        # Initialize ShardListDataset with PIL transformation
        self.dataset = wids.ShardListDataset(shards_info, transformations="PIL")

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return f"tol-10m/{self.split}"

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        sample = self.dataset[i]

        img = next(sample[key] for key in (".jpg", ".png", ".jpeg") if key in sample)
        img = np.array(img)

        # Check if image is monochrome and convert to RGB if needed
        if len(img.shape) == 2:
            self.logger.warning(
                "Monochrome image found in sample '%s' from shard '%s'. Converting to RGB.",
                sample["__key__"],
                sample["__shard__"],
            )
            img = np.stack([img, img, img], axis=2)

        # Check if image has 4 channels (RGBA) and convert to RGB
        elif len(img.shape) == 3 and img.shape[2] == 4:
            self.logger.warning(
                "RGBA image found in sample '%s' from shard '%s'. Converting to RGB by dropping alpha channel.",
                sample["__key__"],
                sample["__shard__"],
            )
            img = img[:, :, :3]  # Keep only the RGB channels, drop the alpha channel

        cls = self.cls_to_idx.get(sample[".taxonomic_name.txt"], -1)
        if cls < 0:
            logger.warning(
                "Missing class '%s' for example '%s' in shard '%s'",
                sample[".taxonomic_name.txt"],
                sample["__key__"],
                sample["__shard__"],
            )

        return img, cls


@beartype.beartype
def _make_tol_class_lookup(
    csv_path: str,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Create a class lookup from a statistics CSV file containing taxonomic information.

    Args:
        stats_path: Path to the statistics CSV file

    Returns:
        Tuple of (cls_to_idx, idx_to_cls) dictionaries
    """
    cls_to_idx = {}
    idx_to_cls = {}

    if not os.path.exists(csv_path):
        logger.warning("Missing csv at '%s'.", csv_path)
        return cls_to_idx, idx_to_cls

    # Define taxonomic columns and their order
    taxonomic_cols = [
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in helpers.progress(reader, desc="Parsing taxonomy", every=1_000_000):
            # Include all taxonomic levels, using empty strings for missing values
            taxa = []
            for col in taxonomic_cols:
                value = row.get(col, "")
                if value.lower() == "nan":
                    value = ""
                taxa.append(value)

            # Create taxonomic string with all levels, including empty ones
            taxonomic_name = " ".join(filter(None, taxa))

            # Add to class mapping if not already present
            if taxonomic_name not in cls_to_idx:
                cls_idx = len(cls_to_idx)
                cls_to_idx[taxonomic_name] = cls_idx
                idx_to_cls[cls_idx] = taxonomic_name

    return cls_to_idx, idx_to_cls


@beartype.beartype
@register_dataset("bioscan-5m")
class Bioscan5M(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        import bioscan_dataset

        self.root = root
        self.split = split
        self.dataset = bioscan_dataset.BIOSCAN5M(root, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return f"bioscan-5m/{self.split}"

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        img, _, cls_idx = self.dataset[i]
        img = np.array(img)
        return img, cls_idx.item()


@beartype.beartype
@register_dataset("fishnet")
class Fishnet(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        import polars as pl

        self.root = root
        self.split = split
        self.csv_file = os.path.join(self.root, f"{split}.csv")
        self.df = pl.read_csv(self.csv_file).with_row_index()
        self.all_columns = [
            "FeedingPath",
            "Tropical",
            "Temperate",
            "Subtropical",
            "Boreal",
            "Polar",
            "freshwater",
            "saltwater",
            "brackish",
        ]
        for col in self.all_columns:
            self.df = self.df.filter(self.df[col].is_not_null())

        # Corresponding column indices
        self.img_col = 4
        self.folder_col = 13
        self.label_col = 5
        logger.info("csv file: %s has %d item.", self.csv_file, len(self.df))

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        row_data = self.df.row(i)

        img_name = row_data[self.img_col]
        img_name = img_name.split("/")[-1]
        folder = row_data[self.folder_col]
        img_path = os.path.join(self.root, "Image_Library", folder, img_name)

        img = cv2.imread(img_path)
        # Change from BGR to RGB.
        img = img[..., ::-1]

        label = row_data[self.label_col]
        if label is None:
            label = -1
        else:
            label = int(label)

        return img, label

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"fishnet/{self.split}"


@beartype.beartype
@register_dataset("newt")
class Newt(Dataset):
    def __init__(self, root: str, split: str, **kwargs):
        import polars as pl

        self.root = root
        self.split = split

        self.labels_csv_path = os.path.join(root, "newt2021_labels.csv")
        if not os.path.isfile(self.labels_csv_path):
            msg = f"Path '{self.labels_csv_path}' doesn't exist. Did you download the Newt dataset?"
            raise RuntimeError(msg)

        # Read the CSV and add row indices
        df = pl.read_csv(self.labels_csv_path).with_row_index()
        # Get all image IDs and labels
        assert split in df.get_column("split").unique().to_list()
        df = df.filter(pl.col("split") == split)
        data = df.select("id", "label").to_numpy(structured=True)
        self.img_ids, self.labels = data["id"], data["label"]

        self.img_dir_path = os.path.join(root, "newt2021_images")

    def __getitem__(self, i: int) -> tuple[UInt8[np.ndarray, "width height 3"], int]:
        img_id = self.img_ids[i]
        img = cv2.imread(os.path.join(self.img_dir_path, f"{img_id}.jpg"))
        # Change from BGR to RGB.
        img = img[..., ::-1]

        label = self.labels[i].item()
        return img, label

    def __len__(self) -> int:
        return len(self.img_ids)

    def __repr__(self) -> str:
        return f"newt/{self.split}"


@beartype.beartype
class TreeOfLife2Scan(Dataset):
    """
    A Parquet-based dataset using incremental chunk reading with PyArrow's `iter_batches`.

    We keep a single iterator (self._batch_iter) that can be advanced as we request higher and higher row indices in the current row group. If the next request is beyond the current iterator's position, we can keep consuming from the iterator without revisiting earlier chunks, thus saving some time.

    However, if the next requested row index is *behind* our current position, or in a different file/row-group, we have to re-initialize the iterator (meaning we start reading from the beginning of that row group again).

    **Tradeoff**:
      - If your data access pattern is mostly forward-sequential or in ascending order, you avoid re-reading early chunks and benefit from caching.
      - If your access pattern is random (sometimes forward, sometimes backward), you might end up resetting the iterator many times, losing the benefit of iterating in chunks.
    """

    def __init__(self, root, split, batch_size=256, **kwargs):
        self.root = root
        self.split = split
        self.files = glob.glob(f"{root}/source={split}/*/*.parquet")
        self.index = []
        for path in self.files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                rcount = pf.metadata.row_group(rg).num_rows
                for r in range(rcount):
                    self.index.append((path, rg, r))

        # Keep track of which file, row-group, and chunk we have loaded.
        self._cached_path = None
        self._cached_rg_idx = None
        self._cached_batch_id = None
        self._cached_batch = None

        # We'll hold the active ParquetFile object and batch iterator.
        self._pf = None
        self._batch_iter = None

        self._batch_size = batch_size

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"treeoflife2/{self.split}"

    def __getitem__(self, i: int):
        if i < 0 or i >= len(self.index):
            raise IndexError

        path, rg_idx, row_idx = self.index[i]
        batch_id = row_idx // self._batch_size
        row_in_chunk = row_idx % self._batch_size

        # If we need a new file or row-group, or if batch_id is behind where we currently are, re-initialize everything.
        need_new_iter = (
            self._pf is None
            or self._cached_path != path
            or self._cached_rg_idx != rg_idx
            or (self._cached_batch_id is not None and batch_id < self._cached_batch_id)
        )
        if need_new_iter:
            self._init_batch_iter(path, rg_idx)
            self._cached_batch_id = -1
            self._cached_batch = None

        # Advance the batch iterator until we reach the requested batch_id
        while self._cached_batch_id < batch_id:
            try:
                self._cached_batch = next(self._batch_iter)
                self._cached_batch_id += 1
            except StopIteration:
                # This means the row_idx is out of range. Should not happen if indexing is correct.
                raise IndexError(
                    f"Failed to load chunk {batch_id} from row-group {rg_idx}"
                )

        # Now self._cached_batch is batch_id. Extract the row we need.
        return self._decode_row(self._cached_batch, row_in_chunk)

    def _init_batch_iter(self, path: str, rg_idx: int):
        """(Re)initialize the iterator and update cached path info."""
        self._pf = pq.ParquetFile(path, memory_map=True)
        # Batches up to batch_size rows each
        self._batch_iter = self._pf.iter_batches(
            row_groups=[rg_idx],
            columns=["image", "original_size", "resized_size"],
            batch_size=self._batch_size,
            use_threads=False,
        )
        self._cached_path = path
        self._cached_rg_idx = rg_idx

    def _decode_row(self, record_batch, row_in_batch: int):
        """Convert the arrow RecordBatch row into (np.ndarray, label)."""
        img_bytes = record_batch.column("image")[row_in_batch].as_py()
        orig_size = record_batch.column("original_size")[row_in_batch].as_py()
        resized_size = record_batch.column("resized_size")[row_in_batch].as_py()

        # Convert to NumPy array
        N_CHANNELS = 3
        np_image = np.frombuffer(img_bytes, dtype=np.uint8)

        for dims in [orig_size, resized_size]:
            if dims and isinstance(dims, list) and len(dims) >= 2:
                height, width = dims[:2]
                expected_size = height * width * N_CHANNELS
                if np_image.size == expected_size:
                    # Convert from BGR to RGB
                    return np_image.reshape((height, width, N_CHANNELS))[..., ::-1], -1

        logging.warning("Image size does not match expected dimensions.")
        return None, -1


@beartype.beartype
class TreeOfLife2Binary(Dataset):
    """
    A Parquet-based dataset using incremental chunk reading with PyArrow's `iter_batches`.

    We avoid storing a per-row index (which can be huge) by only keeping row-group boundaries.
    For an index i, we binary-search to find which row group contains i, and then compute the
    offset-in-row-group.

    We keep a single iterator (self._batch_iter) that can be advanced as we request higher
    and higher row indices in the current row group. If the next request is beyond the current
    iterator's position, we can keep consuming from the iterator without revisiting earlier
    chunks, thus saving time.

    If the next requested row index is behind our current position, or in a different file/row
    group, we have to re-initialize the iterator (meaning we start reading from the beginning
    of that row group again).

    **Tradeoff**:
      - If your data access pattern is mostly forward-sequential or in ascending order, you avoid
        re-reading early chunks and benefit from caching.
      - If your access pattern is random (sometimes forward, sometimes backward), you might end
        up resetting the iterator many times, losing the benefit of iterating in chunks.
    """

    def __init__(self, root, split, batch_size=256, **kwargs):
        self.root = root
        self.split = split
        self.files = glob.glob(f"{root}/source={split}/*/*.parquet")

        # We'll store row-group boundaries and associated file+rg indices.
        self._rg_paths = []
        self._rg_indices = []
        self._rg_starts = []
        self._rg_ends = []
        self._total_len = 0

        for path in helpers.progress(
            self.files, desc="pfs", every=len(self.files) // 100 + 1
        ):
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                rcount = pf.metadata.row_group(rg).num_rows
                start = self._total_len
                end = start + rcount
                self._rg_paths.append(path)
                self._rg_indices.append(rg)
                self._rg_starts.append(start)
                self._rg_ends.append(end)
                self._total_len = end

        # Keep track of which file, row-group, and chunk we have loaded.
        self._cached_path = None
        self._cached_rg_idx = None
        self._cached_batch_id = None
        self._cached_batch = None

        # We'll hold the active ParquetFile object and batch iterator.
        self._pf = None
        self._batch_iter = None

        self._batch_size = batch_size

    def __len__(self) -> int:
        return self._total_len

    def __repr__(self) -> str:
        return f"treeoflife2/{self.split}"

    def __getitem__(self, i: int):
        if i < 0 or i >= self._total_len:
            raise IndexError(f"Index out of range: {i} (length {self._total_len})")

        # Find which row-group contains i using a binary search in _rg_ends
        rg_idx = bisect.bisect_right(self._rg_ends, i)
        # offset within this row group
        row_in_rg = i - self._rg_starts[rg_idx]

        path = self._rg_paths[rg_idx]
        row_group_num = self._rg_indices[rg_idx]

        batch_id = row_in_rg // self._batch_size
        row_in_chunk = row_in_rg % self._batch_size

        # If we need a new file/rg or if batch_id is behind the cached position, re-init
        need_new_iter = (
            self._pf is None
            or self._cached_path != path
            or self._cached_rg_idx != row_group_num
            or (self._cached_batch_id is not None and batch_id < self._cached_batch_id)
        )
        if need_new_iter:
            self._init_batch_iter(path, row_group_num)
            self._cached_batch_id = -1
            self._cached_batch = None

        # Advance the batch iterator until we reach the requested batch_id
        while self._cached_batch_id < batch_id:
            try:
                self._cached_batch = next(self._batch_iter)
                self._cached_batch_id += 1
            except StopIteration:
                raise IndexError(
                    f"Failed to load chunk {batch_id} from row-group {row_group_num}"
                )

        # Now self._cached_batch is batch_id. Extract the row we need.
        return self._decode_row(self._cached_batch, row_in_chunk)

    def _init_batch_iter(self, path: str, rg_idx: int):
        """(Re)initialize the iterator and update cached path info."""
        self._pf = pq.ParquetFile(path, memory_map=True)
        self._batch_iter = self._pf.iter_batches(
            row_groups=[rg_idx],
            columns=["image", "original_size", "resized_size"],
            batch_size=self._batch_size,
            use_threads=False,
        )
        self._cached_path = path
        self._cached_rg_idx = rg_idx

    def _decode_row(self, record_batch, row_in_batch: int):
        """Convert the arrow RecordBatch row into (np.ndarray, label)."""
        img_bytes = record_batch.column("image")[row_in_batch].as_py()
        orig_size = record_batch.column("original_size")[row_in_batch].as_py()
        resized_size = record_batch.column("resized_size")[row_in_batch].as_py()

        # Convert to NumPy array
        N_CHANNELS = 3
        np_image = np.frombuffer(img_bytes, dtype=np.uint8)

        for dims in [orig_size, resized_size]:
            if dims and isinstance(dims, list) and len(dims) >= 2:
                height, width = dims[:2]
                expected_size = height * width * N_CHANNELS
                if np_image.size == expected_size:
                    # Convert from BGR to RGB
                    return np_image.reshape((height, width, N_CHANNELS))[..., ::-1], -1

        logging.warning("Image size does not match expected dimensions.")
        return None, -1


@beartype.beartype
class TreeOfLife2Fragments(Dataset):
    """
    A PyArrow.dataset-based Parquet dataset in which each file has exactly one row group.

    We keep track of each file fragment's row count (rather than row-group counts).
    For random access, we do:
      - Binary search on fragment boundaries.
      - Within the chosen fragment, use a chunk iterator (batch_size).
      - If the requested index is behind our current chunk read position, or in a
        different fragment, we reset the iterator.
    """

    def __init__(self, root, split, batch_size=256, **kwargs):
        self.root = root
        self.split = split
        self._batch_size = batch_size

        # Build list of Parquet files. Then let Arrow handle them as a unified Dataset.
        parquet_files = glob.glob(f"{root}/source={split}/*/*.parquet")
        self._dataset = ds.dataset(parquet_files, format="parquet")

        # Each fragment should correspond to exactly one row group. We'll store:
        #  self._fragments[i]: the ith fragment
        #  self._frag_starts[i]: inclusive start index
        #  self._frag_ends[i]: exclusive end index
        # so that global index i belongs to the fragment f where frag_starts[f] <= i < frag_ends[f].
        self._fragments = []
        self._frag_starts = []
        self._frag_ends = []
        self._total_len = 0

        # Enumerate all fragments. We rely on each having exactly one row group.
        fragments = self._dataset.get_fragments()
        for fragment in helpers.progress(fragments, desc="pfs", every=200):
            # Ensure we have metadata so we can see the row count.
            if fragment.metadata is None:
                fragment.ensure_complete_metadata()

            md = fragment.metadata  # pyarrow.parquet.FileMetaData
            assert md.num_row_groups == 1, (
                f"Expected exactly one row group per parquet file, but found {md.num_row_groups} row groups in fragment: {fragment.path}"
            )

            # There's exactly one row group. Get its row count.
            row_count = md.row_group(0).num_rows

            start = self._total_len
            end = start + row_count

            self._fragments.append(fragment)
            self._frag_starts.append(start)
            self._frag_ends.append(end)
            self._total_len = end

        # Cache for the current active fragment and batch
        self._cached_fragment = None
        self._cached_batch_id = None
        self._cached_batch = None
        self._batch_iter = None

    def __len__(self) -> int:
        return self._total_len

    def __repr__(self) -> str:
        return f"treeoflife2fragments/{self.split}"

    def __getitem__(self, i: int):
        if i < 0 or i >= self._total_len:
            raise IndexError(f"Index {i} out of range (0..{self._total_len - 1})")

        # Find which fragment holds this global index i
        frag_idx = bisect.bisect_right(self._frag_ends, i)
        offset_in_fragment = i - self._frag_starts[frag_idx]

        fragment = self._fragments[frag_idx]

        batch_id = offset_in_fragment // self._batch_size
        row_in_chunk = offset_in_fragment % self._batch_size

        # If we switched fragments or are jumping backward in the same fragment, re-init
        need_new_iter = (
            self._cached_fragment is None
            or self._cached_fragment is not fragment
            or (self._cached_batch_id is not None and batch_id < self._cached_batch_id)
        )
        if need_new_iter:
            self._init_batch_iter(fragment)
            self._cached_batch_id = -1
            self._cached_batch = None

        # Advance until we reach batch_id
        while self._cached_batch_id < batch_id:
            try:
                self._cached_batch = next(self._batch_iter)
                self._cached_batch_id += 1
            except StopIteration:
                raise IndexError(f"Index {i} not found in fragment {frag_idx}?")

        # Now extract row_in_chunk from self._cached_batch
        return self._decode_row(self._cached_batch, row_in_chunk)

    def _init_batch_iter(self, fragment: ds.Fragment):
        """(Re)initialize the batch iterator for this single-fragment Parquet file."""
        self._cached_fragment = fragment
        self._batch_iter = fragment.to_batches(
            batch_size=self._batch_size,
            columns=["image", "original_size", "resized_size"],
            use_threads=False,
        )

    def _decode_row(self, record_batch: pa.RecordBatch, row_in_batch: int):
        """Convert the arrow RecordBatch row into (np.ndarray, label)."""
        img_bytes = record_batch.column("image")[row_in_batch].as_py()
        orig_size = record_batch.column("original_size")[row_in_batch].as_py()
        resized_size = record_batch.column("resized_size")[row_in_batch].as_py()

        # Convert to NumPy array
        np_image = np.frombuffer(img_bytes, dtype=np.uint8)
        N_CHANNELS = 3

        for dims in [orig_size, resized_size]:
            if dims and isinstance(dims, list) and len(dims) >= 2:
                height, width = dims[:2]
                expected_size = height * width * N_CHANNELS
                if np_image.size == expected_size:
                    # BGR -> RGB
                    return np_image.reshape((height, width, N_CHANNELS))[:, :, ::-1], -1

        logging.warning("Image size does not match expected dimensions.")
        return None, -1


@beartype.beartype
@register_dataset("treeoflife2")
class TreeOfLife2Concurrent(Dataset):
    """
    ThreadPool-based Parquet metadata loading. Reduces latency by parallelizing Parquet file
    metadata reads. Useful when many Parquet files exist.
    """

    def __init__(self, root, split, batch_size=256, max_workers=16, **kwargs):
        self.root = root
        self.split = split
        self.files = glob.glob(f"{root}/source={split}/*/*.parquet")
        self._batch_size = batch_size

        self._rg_paths = []
        self._rg_indices = []
        self._rg_starts = []
        self._rg_ends = []
        self._total_len = 0

        def load_rg_metadata(path):
            pf = pq.ParquetFile(path)
            n_rows = [
                pf.metadata.row_group(rg).num_rows for rg in range(pf.num_row_groups)
            ]
            return path, n_rows

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_rg_metadata, path) for path in self.files]

            for future in helpers.progress(
                futures, desc="loading metadata", every=len(futures) // 20 + 1
            ):
                path, num_rows = future.result()
                for rg_idx, rcount in enumerate(num_rows):
                    start = self._total_len
                    end = start + rcount
                    self._rg_paths.append(path)
                    self._rg_indices.append(rg_idx)
                    self._rg_starts.append(start)
                    self._rg_ends.append(end)
                    self._total_len = end

        # Cache state for batch iterator
        self._cached_path = None
        self._cached_rg_idx = None
        self._cached_batch_id = None
        self._cached_batch = None

        self._pf = None
        self._batch_iter = None

    def __len__(self) -> int:
        return self._total_len

    def __repr__(self) -> str:
        return f"treeoflife2/{self.split}"

    def __getitem__(self, i: int):
        if i < 0 or i >= self._total_len:
            raise IndexError(f"Index out of range: {i} (length {self._total_len})")

        rg_idx = bisect.bisect_right(self._rg_ends, i)
        row_in_rg = i - self._rg_starts[rg_idx]

        path = self._rg_paths[rg_idx]
        row_group_num = self._rg_indices[rg_idx]

        batch_id = row_in_rg // self._batch_size
        row_in_chunk = row_in_rg % self._batch_size

        need_new_iter = (
            self._pf is None
            or self._cached_path != path
            or self._cached_rg_idx != row_group_num
            or (self._cached_batch_id is not None and batch_id < self._cached_batch_id)
        )

        if need_new_iter:
            self._init_batch_iter(path, row_group_num)
            self._cached_batch_id = -1
            self._cached_batch = None

        while self._cached_batch_id < batch_id:
            try:
                self._cached_batch = next(self._batch_iter)
                self._cached_batch_id += 1
            except StopIteration:
                raise IndexError(
                    f"Failed to load chunk {batch_id} from row-group {row_group_num}"
                )

        return self._decode_row(self._cached_batch, row_in_chunk)

    def _init_batch_iter(self, path: str, rg_idx: int):
        self._pf = pq.ParquetFile(path, memory_map=True)
        self._batch_iter = self._pf.iter_batches(
            row_groups=[rg_idx],
            columns=["image", "original_size", "resized_size"],
            batch_size=self._batch_size,
            use_threads=False,
        )
        self._cached_path = path
        self._cached_rg_idx = rg_idx

    def _decode_row(self, record_batch, row_in_batch: int):
        img_bytes = record_batch.column("image")[row_in_batch].as_py()
        orig_size = record_batch.column("original_size")[row_in_batch].as_py()
        resized_size = record_batch.column("resized_size")[row_in_batch].as_py()

        N_CHANNELS = 3
        np_image = np.frombuffer(img_bytes, dtype=np.uint8)

        for dims in [orig_size, resized_size]:
            if dims and isinstance(dims, list) and len(dims) >= 2:
                height, width = dims[:2]
                expected_size = height * width * N_CHANNELS
                if np_image.size == expected_size:
                    return np_image.reshape((height, width, N_CHANNELS))[..., ::-1], -1

        logging.warning("Image size does not match expected dimensions.")
        return None, -1


@beartype.beartype
def load(key: str, split: str, root: str, **kwargs) -> Dataset:
    """Load a dataset by key."""
    # Check if the key is in the registry
    if key in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[key](root, split, **kwargs)

    raise ValueError(
        f"Dataset '{key}' not found. Available datasets: {list_datasets()}"
    )
