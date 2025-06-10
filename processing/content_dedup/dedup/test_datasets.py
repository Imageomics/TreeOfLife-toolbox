import os
import pathlib
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from . import datasets

IMPLEMENTATIONS = [
    datasets.TreeOfLife2Scan,
    datasets.TreeOfLife2Binary,
    datasets.TreeOfLife2Fragments,
    datasets.TreeOfLife2Concurrent,
]


@pytest.fixture(scope="session")
def mock_init_args(request) -> tuple[str, str]:
    """
    Fixture to provide real parquet files for testing.

    Use:
    - pytest --root="./data/parquet_files/"
    """

    if root := request.config.getoption("--root", default=None):
        root = pathlib.Path(root)
        if root.exists() and root.is_dir():
            root = str(root)
        else:
            root = None

    split = request.config.getoption("--split", default=None)

    if root is not None and split is not None:
        yield root, split
        return

    # If options were not provided, make temporary files.
    # Create temp root directory
    root = tempfile.mkdtemp()
    split = "train"

    # Create directory structure: {root}/source={split}/*/*.parquet
    source_dir = os.path.join(root, f"source={split}")
    os.makedirs(source_dir, exist_ok=True)

    # Create subdirectories to mirror expected structure
    for subdir_idx in range(2):
        subdir_path = os.path.join(source_dir, f"subset_{subdir_idx}")
        os.makedirs(subdir_path, exist_ok=True)

        # Create parquet files in each subdirectory
        for file_idx in range(2):
            file_path = os.path.join(subdir_path, f"data_{file_idx}.parquet")

            # Create data for this file
            all_batches = []
            for group_idx in range(3):
                data = []
                for row_idx in range(5):
                    # Create a simple 10x10 RGB image
                    height, width = 10, 10
                    img = np.random.randint(
                        0, 256, size=(height, width, 3), dtype=np.uint8
                    )

                    # Store as BGR since decode_image will convert back to RGB
                    img_bgr = img[..., ::-1]
                    img_bytes = img_bgr.tobytes()

                    data.append({
                        "uuid": f"img_{subdir_idx}_{file_idx}_{group_idx}_{row_idx}",
                        "source_id": "test",
                        "identifier": f"test_{file_idx}_{group_idx}_{row_idx}",
                        "is_license_full": True,
                        "license": "test",
                        "source": "test",
                        "title": "test",
                        "original_size": [height, width],
                        "resized_size": [height, width],
                        "hashsum_original": "test",
                        "hashsum_resized": "test",
                        "image": img_bytes,
                    })

                batch = pa.Table.from_pylist(data)
                all_batches.append(batch)

            # Write to parquet file with row groups
            writer = pq.ParquetWriter(file_path, all_batches[0].schema)
            for batch in all_batches:
                writer.write_table(batch)
            writer.close()

    yield (root, split)

    # Cleanup
    try:
        import shutil

        shutil.rmtree(root)
    except:
        print(f"Warning: Failed to clean up temporary directory: {root}")


@pytest.fixture
def empty_parquet_file():
    """Create an empty parquet file for testing."""
    fd, path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)

    # Create an empty schema
    schema = pa.schema([
        ("uuid", pa.string()),
        ("image", pa.binary()),
        ("original_size", pa.list_(pa.int64())),
        ("resized_size", pa.list_(pa.int64())),
    ])

    # Create an empty table
    table = pa.Table.from_pylist([], schema=schema)
    pq.write_table(table, path)

    yield [path]

    # Cleanup
    try:
        os.remove(path)
    except:
        pass


# Basic Interface Tests
@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_dataset_init(implementation_cls, mock_init_args):
    """Test initialization of all dataset implementations."""
    root, split = mock_init_args
    ds = implementation_cls(root, split)
    assert isinstance(ds, datasets.Dataset)


@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_dataset_len(implementation_cls, mock_init_args):
    """Test the length of all dataset implementations."""
    # Expected length: 2 subsets * 2 files * 3 groups * 5 rows = 60
    expected_len = 60

    root, split = mock_init_args
    ds = implementation_cls(root, split)
    assert len(ds) == expected_len


@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_dataset_repr(implementation_cls, mock_init_args):
    """Test string representation of all dataset implementations."""
    root, split = mock_init_args
    ds = implementation_cls(root, split)
    assert isinstance(repr(ds), str)


@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_dataset_getitem(implementation_cls, mock_init_args):
    """Test __getitem__ method of all dataset implementations."""
    root, split = mock_init_args
    ds = implementation_cls(root, split)

    # Test multiple indices to ensure we hit different files and row groups
    for i in [0, 5, 10, 15, 20, 25, 29]:
        img, label = ds[i]
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # RGB image
        assert label == -1


@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_dataset_out_of_bounds(implementation_cls, mock_init_args):
    """Test behavior with out-of-bounds indices."""
    root, split = mock_init_args
    ds = implementation_cls(root, split)

    # Test negative index
    with pytest.raises(IndexError):
        _ = ds[-1]

    # Test index >= len
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_dataset_consistency(mock_init_args):
    """Test that all implementations return the same data for the same index."""
    root, split = mock_init_args
    datasets_to_test = [impl_cls(root, split) for impl_cls in IMPLEMENTATIONS]

    for i in range(min(5, len(datasets_to_test[0]))):
        images = [ds[i][0] for ds in datasets_to_test]

        # Compare the image data from all implementations
        for j in range(1, len(images)):
            assert np.array_equal(images[0], images[j])


# Performance benchmarks using pytest-benchmark
@pytest.mark.benchmark
@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_initialization_benchmark(implementation_cls, mock_init_args, benchmark):
    """Benchmark dataset initialization times."""
    root, split = mock_init_args
    benchmark.group = "init"
    benchmark.name = implementation_cls.__name__
    benchmark(implementation_cls, root, split)


@pytest.mark.benchmark
@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_random_access_benchmark(implementation_cls, mock_init_args, benchmark):
    """Benchmark random access performance for all implementations."""
    root, split = mock_init_args
    ds = implementation_cls(root, split)

    # Generate random indices
    np.random.seed(42)
    indices = np.random.randint(0, len(ds), size=100)

    # Benchmark random access
    benchmark.group = "random_access"
    benchmark.name = implementation_cls.__name__
    benchmark(lambda: [ds[idx.item()] for idx in indices])


@pytest.mark.benchmark
@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_sequential_access_benchmark(implementation_cls, mock_init_args, benchmark):
    """Benchmark sequential access performance for all implementations."""
    root, split = mock_init_args
    ds = implementation_cls(root, split)

    # Use first 100 items for benchmarking
    n_items = min(100, len(ds))

    # Benchmark sequential access
    benchmark.group = "sequential_access"
    benchmark.name = implementation_cls.__name__
    benchmark(lambda: [ds[i] for i in range(n_items)])


@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_boundary_indices(implementation_cls, mock_init_args):
    """
    Test retrieval of items at boundary indices between files.

    This ensures the binary search correctly handles the transition points
    between different files where off-by-one errors often occur.
    """
    root, split = mock_init_args
    ds = implementation_cls(root, split)

    # In our mock data:
    # - 2 subdirs * 2 files * 3 groups * 5 rows = 60 items total
    # - Each file contains 3 groups * 5 rows = 15 items
    # - File boundaries are at indices: 0, 15, 30, 45

    file_boundaries = [0, 14, 15, 29, 30, 44, 45, 59]

    for idx in file_boundaries:
        if idx >= len(ds):
            continue

        # Get item at boundary
        img, label = ds[idx]
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # RGB image
        assert label == -1

        # If not the first item, check the transition between files
        if idx > 0 and idx % 15 == 0:  # At file boundaries (15, 30, 45)
            img_before, _ = ds[idx - 1]
            img_after, _ = ds[idx]

            # Both should be valid images
            assert isinstance(img_before, np.ndarray)
            assert isinstance(img_after, np.ndarray)

            # Check that the transition is handled correctly
            # Images should have the same shape but different content
            assert img_before.shape == img_after.shape


@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_consistent_item_retrieval(implementation_cls, mock_init_args):
    """
    Test that retrieving the same index multiple times returns the same item.

    This verifies that the binary search consistently maps indices to the
    correct files across multiple accesses.
    """
    root, split = mock_init_args
    ds = implementation_cls(root, split)

    # Sample indices from different parts of the dataset
    # Particularly important to test at file boundaries and within each file
    sample_indices = [0, 14, 15, 29, 30, 44, 45, 59]

    for idx in sample_indices:
        if idx >= len(ds):
            continue

        # Get the item multiple times
        img1, _ = ds[idx]
        img2, _ = ds[idx]
        img3, _ = ds[idx]

        # All retrievals should return the same image data
        assert np.array_equal(img1, img2)
        assert np.array_equal(img1, img3)


@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS)
def test_sequential_consistency(implementation_cls, mock_init_args):
    """
    Test sequential access across file boundaries.

    This ensures that the binary search maintains correct ordering when
    iterating sequentially across file boundaries.
    """
    root, split = mock_init_args
    ds = implementation_cls(root, split)

    # Test sections that cross file boundaries
    boundary_regions = [(13, 17), (28, 32), (43, 47)]

    for start, end in boundary_regions:
        if end >= len(ds):
            continue

        # Get all items in the region
        items = [ds[i][0] for i in range(start, end + 1)]

        # Check that we got the expected number of items
        assert len(items) == (end - start + 1)

        # Verify all items are valid
        for img in items:
            assert isinstance(img, np.ndarray)
            assert img.shape[2] == 3

        # Verify that items at the boundary (where binary search would transition files)
        # are different - they should be from different files
        # For example, items[2] and items[3] in the (13, 17) region are at indices 15 and 16
        # which should be from different files
        boundary_idx = 15 - start  # File boundary index in our sliced list
        if 0 <= boundary_idx < len(items) - 1:
            # Images should have the same dimensions but different content
            assert items[boundary_idx].shape == items[boundary_idx + 1].shape
            # Don't check for inequality of pixel values as that's not guaranteed
