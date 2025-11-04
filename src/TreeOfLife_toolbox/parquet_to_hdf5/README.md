# Parquet to HDF5 Conversion Tool

This tool converts TreeOfLife Parquet files containing raw image bytes into a hybrid data format consisting of:
- HDF5 files with UUID-indexed lossless WebP compressed images
- Separate Parquet metadata files (without image data)

The tool uses hardcoded WebP lossless compression with method=6 for optimal space efficiency while preserving image quality.

## How to Access and Use the Converted Data

### Output Structure

After conversion, each input Parquet file produces two output files:

```
<output_directory>/
├── <filename>_images.h5      # HDF5 file containing compressed images
└── <filename>_metadata.parquet   # Parquet file with all metadata (no images)
```

### Accessing Images from HDF5 Files

The HDF5 files store WebP-compressed images indexed by their UUIDs. Here are efficient access patterns:

#### Basic Access with Random Selection

```python
import io, random
from pathlib import Path
import polars as pl
import h5py
from PIL import Image

H5_PATH   = Path("filename_images.h5")
META_PATH = Path("filename_metadata.parquet")

# Get UUIDs from metadata
uuids = pl.read_parquet(META_PATH)["uuid"].to_list()
print(f"Available UUIDs: {len(uuids)}")

# Open HDF5 and access file attributes
with h5py.File(H5_PATH, "r") as h5f:
    print(f"Total images: {h5f.attrs.get('total_images')}")
    cr = h5f.attrs.get("compression_ratio", 0.0)
    print(f"Compression ratio: {cr:.1f}:1")
    print(f"Image format: {h5f.attrs.get('image_format')}")

    # Pick a random UUID and fetch its WebP data
    uuid = random.choice(uuids)
    data = h5f["images"][uuid][()]  # Raw WebP byte stream

    # Display image (decoded in-memory)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.show()

    # Save exact original WebP (bit-identical, no re-encoding)
    with open(f"{uuid}.webp", "wb") as f:
        f.write(data)
    print(f"Saved bit-identical {uuid}.webp")
```

#### Fetching Specific UUIDs

```python
import io
import polars as pl
import h5py
from PIL import Image

H5_PATH   = "filename_images.h5"
META_PATH = "filename_metadata.parquet"

# Get specific UUIDs (example: first 10, or your own list)
wanted_uuids = pl.read_parquet(META_PATH)["uuid"][:10].to_list()
print(f"Requesting {len(wanted_uuids)} images")

with h5py.File(H5_PATH, "r") as h5f:
    images = h5f["images"]
    for uuid in wanted_uuids:
        if uuid in images:  # Direct lookup
            data = images[uuid][()]  # Raw WebP bytes

            # Decode for processing/inspection
            img = Image.open(io.BytesIO(data)).convert("RGB")

            # Save with lossless WebP re-encoding (preserves quality)
            # NOTE! This save method is not guaranteed to be bit-identical for matching checksums! (re-encoding step)
            img.save(f"{uuid}.webp", format="WebP", lossless=True, method=6)
        else:
            print(f"Missing in HDF5: {uuid}")
```

### Accessing Metadata from Parquet Files

```python
import polars as pl

# Load metadata (contains all original columns except 'image')
metadata_df = pl.read_parquet('filename_metadata.parquet')

# View available columns
print(metadata_df.columns)

# Filter for specific metadata
uuid_to_find = "your-uuid-here"
image_metadata = metadata_df.filter(pl.col('uuid') == uuid_to_find)
print(image_metadata)
```

### File Format Specifications

**HDF5 Structure:**
- Root attributes: `total_images`, `compression_ratio`, `image_format`, etc.
- `images/` group: Contains datasets named by UUID
- Each dataset: Compressed WebP image as numpy uint8 array
- WebP settings: Lossless compression, method=6, quality=100

**Parquet Structure:**
- Contains all original columns from source Parquet except `image`
- Compressed with zstd for efficient storage
- Maintains original data types and schema