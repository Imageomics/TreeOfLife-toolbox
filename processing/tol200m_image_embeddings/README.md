# TreeOfLife-200M Image Embeddings

Provenance and reproduction scripts for the image-embedding configs published in
[imageomics/TreeOfLife-200M-Embeddings](https://huggingface.co/datasets/imageomics/TreeOfLife-200M-Embeddings):

| Config | Model | Dim | Stored precision | Normalized |
|--------|-------|-----|------------------|------------|
| `bioclip-2_float16` | [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) (ViT-L/14) | 768 | float16 | No |
| `bioclip-2.5-vith14_float16` | [BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14) (ViT-H/14) | 1024 | float16 | Yes (L2) |

Both models run through the **same pipeline**, differing only in model weights (and the
normalization decision at standardization time). The pipeline has two stages:

```
TreeOfLife-200M images (HDF5 shards)
        |  1. batch embedding  (hpc-inference, GPU SLURM jobs)
        v
raw per-rank parquet  (uuid, emb_<model> float32)
        |  2. standardization  (CPU SLURM job: catalog metadata join, taxonomic
        |     sort, L2-normalization if applicable, float16 cast, contract write)
        v
train-NNNNN-of-MMMMM.parquet  (published config layout)
```

Dataset layout, column semantics, sort order, and the parquet contract are documented in
the [dataset card](https://huggingface.co/datasets/imageomics/TreeOfLife-200M-Embeddings);
this README stays operational and does not repeat them.

## Environment

```bash
uv pip install "hpc-inference[openclip] @ git+https://github.com/Imageomics/hpc-inference.git"
uv pip install duckdb pyarrow h5py nvidia-ml-py
```

## Stage 1: batch embedding

Input is the TreeOfLife-200M image collection materialized as HDF5 shards
(`<DATA_ROOT>/source=<s>/server=<y>/*_images.h5`, one `uuid -> image bytes` entry per
image, with a paired `*_metadata.parquet` per shard). Embedding uses the
[hpc-inference](https://github.com/Imageomics/hpc-inference) package's
`open_clip_embed` entry point (`--input_type hdf5`); each SLURM task is one GPU rank,
and ranks split the shard list automatically.

For sources with a moderate file count (bioscan, eol, fathomnet: 14-654 files),
submit one multi-rank job. Keep `world_size <= num_files` and ideally
`files_per_rank >= num_workers`:

```bash
export RUN_CONFIG=configs/embed_bioclip_2_5.yaml       # or embed_bioclip_2.yaml
export RUN_TARGET_DIR=<DATA_ROOT>/source=bioscan
export RUN_OUTPUT_DIR=<EMB_OUT>/model=bioclip_2_5/source=bioscan
sbatch --nodes=4 --time=02:00:00 scripts/embed.slurm   # 4 nodes x 1 GPU = 4 ranks
```

For sources with many files of highly variable size (gbif: ~24.6K files, 1-20K
images each), first build image-count-balanced chunks, then submit a SLURM array
of single-GPU tasks. Balancing chunks by image count (not file count) is what
keeps task walltimes even:

```bash
python scripts/count_images.py <DATA_ROOT>/source=gbif file_counts.parquet
NTASKS=$(python scripts/build_chunks.py file_counts.parquet \
    <EMB_OUT>/model=bioclip_2_5/source=gbif 7000000)

export RUN_CONFIG=configs/embed_bioclip_2_5.yaml
export RUN_TARGET_DIR=<DATA_ROOT>/source=gbif
export RUN_OUT_ROOT=<EMB_OUT>/model=bioclip_2_5/source=gbif
sbatch --array=0-$((NTASKS-1))%6 --time=06:00:00 scripts/embed_array.slurm
```

Both paths write `.../embeddings/rank_<R>/<prefix>_rank_<R>_<i>.parquet` with columns
`uuid` + `emb_<model_key>` (raw float32, un-normalized). Budget the array `--time`
for the worst chunk: chunks packing many small files run slower per image because
of per-file HDF5 open overhead.

> **Precision note.** The published `bioclip-2.5-vith14_float16` production run used
> bf16 autocast (plus in-pipeline normalization / float16 storage) for
> throughput gain; those pipeline features are not yet in the released `hpc-inference`
> package, so these scripts run the released fp32 path and defer normalization and
> casting to Stage 2. The result is equivalent up to compute precision: measured
> cosine similarity between bf16- and fp32-computed embeddings is 0.9998 mean
> (min 0.993), and after L2 normalization the stored float16 cast is lossless.

## Stage 2: standardization

Turns raw per-rank embeddings into a published config. Metadata is **not** re-derived:
every config shares the exact row set, row order, and 15 metadata columns of the
already-published `bioclip-2_float16` config, which is itself the
[TreeOfLife-200M catalog](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
(revision `94bbc0b`) restricted to its 233,055,986 rows and globally sorted by
```
source_dataset > 
kingdom > phylum > class > order > family > genus > species > common_name
```
Reusing the published row order (rather than re-sorting the catalog) keeps all configs
row-aligned for free: the sort keys have large tie groups, so a fresh sort would not
reproduce the published order within ties.

First build the spine (`uuid` + metadata + global `_pos` index, stripped from the
published config files, order preserved). Run once; every config build reuses it:

```bash
python scripts/build_spine.py <PUBLISHED_DIR> <SPINE_DIR>
```

Then build the config. DuckDB inner-joins the spine to the raw embeddings by `uuid`
(dropping raw rows not in the TOL catalog), orders by `_pos`, and a parallel writer pool
applies L2 normalization in float32 (`--normalize`, used for bioclip-2.5, not for
bioclip-2), casts to `fixed_size_list<float16>[dim]`, and writes the contract
parquet files:

```bash
python scripts/standardize_config.py emb_bioclip_2_5 bioclip-2.5-vith14_float16 250000 \
    --dim 1024 --normalize \
    --spine "<SPINE_DIR>/train-*.parquet" \
    --emb-glob "<EMB_OUT>/model=bioclip_2_5/source=*/embeddings/rank_*/*.parquet" \
    --emb-glob "<EMB_OUT>/model=bioclip_2_5/source=gbif/task_*/embeddings/rank_*/*.parquet" \
    --out <STAGE_DIR>

# bioclip-2 equivalent (no --normalize; 768-dim; 500K rows/file):
# python scripts/standardize_config.py emb_bioclip_2 bioclip-2_float16 500000 \
#     --dim 768 --spine ... --emb-glob ... --out <STAGE_DIR>
```

`rows_per_file` targets ~500 MB files: 500000 rows per file for 512/768-dim, 250000 for 1024-dim.
Run as a SLURM cpu job; the global join+sort spills through DuckDB temp space
(~470 GB peak for the 1024-dim config, so use a hugemem-class node; ~340 GB for
512-dim).

## Stage 3: validation

Checks a finished config: file / row / distinct-uuid counts, parquet contract fields,
FULL uuid-at-position order match against the spine, and a sample recompute of the
normalize+cast transform against the raw embeddings:

```bash
python scripts/validate_config.py <STAGE_DIR>/bioclip-2.5-vith14_float16 emb_bioclip_2_5 \
    --dim 1024 --normalize \
    --spine "<SPINE_DIR>/train-*.parquet" \
    --emb-glob "<EMB_OUT>/model=bioclip_2_5/source=*/embeddings/rank_*/*.parquet" \
    --emb-glob "<EMB_OUT>/model=bioclip_2_5/source=gbif/task_*/embeddings/rank_*/*.parquet"
```

Exit code 0 and `VERDICT: PASS` mean the config is ready to upload.

## Placeholders

All `scripts/configs` use placeholders; set them before running:

| Placeholder | Meaning |
|---|---|
| `YOUR_ACCOUNT` | SLURM account |
| `/path/to/venv` | Python venv with the environment above |
| `<DATA_ROOT>` | TreeOfLife-200M HDF5 shard root (`source=<s>/...`) |
| `<EMB_OUT>` | raw embedding output root (scratch) |
| `<PUBLISHED_DIR>` | local copy of the published `bioclip-2_float16/` files |
| `<SPINE_DIR>` | spine output dir (intermediate, reusable) |
| `<STAGE_DIR>` | standardized config output root (upload staging) |
