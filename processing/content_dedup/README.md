# Content-Based Deduplication

We want to implement perceptual hashing to prevent train-test leakage.
[Perceptual hashes](https://en.wikipedia.org/wiki/Perceptual_hashing) are a type of hash function that is resistant to modifications of the input, as compared to hashes like SHA-1 or MD5.
However, we need to pick a hash algorithm and its settings so that we don't remove false positives, or miss true positives.

**How do we pick a hash algorithm and its settings?**

We are evaluating [PDQ](https://github.com/facebook/ThreatExchange/tree/main/pdq) ([whitepaper](https://github.com/facebook/ThreatExchange/blob/main/hashing/hashing.pdf)) and [pHash](https://www.phash.org/).

The only setting to pick is the "match threshold", which is how close two hashes can be before we consider the images to be identical.

To decide between PDQ and pHash, and to tune the match threshold, we do two experiments.
Results are at [`./experiments/exp1.pdf`](./experiments/exp1.pdf) and [`./experiments/exp2.pdf`](./experiments/exp2.pdf).

While each report contains an experimental methodology, we briefly describe the methodology here as well.

## Experiment #1

This experiment develops intuition around how images cluster within training splits for different algorithms and thresholds.

1. For each algorithm (PDQ, pHash) and each dataset (iNat2021/train, ToL-10M/train, ToL-2, BioSCAN-5M/train), hash all of the images in the dataset.
2. Within each algorithm/dataset pair, sample 100K pairs of hashes and record their distance.
3. Plot a histogram of these distances.
4. Sample 10 different thresholds among these distances. It's still not decided how to sample these 10 different thresholds (uniformly from $[min, max]$? Just randomly from all distances?).
5. For each threshold, randomly sample 10 images, then get all the images within the dataset that match those "seed" images.
6. Record these images to develop human intuition for what different thresholds and algorithms lead to.

The outputs are step #3 and #6.

## Experiment #2

This experiment develops intuition for what kinds of images are filtered if we use perceptual hashes to remove test images from our training data.

1. For each algorithm (PDQ, pHash), and each dataset (iNat2024/test, Fishnet/test, NeWT/test) hash 100 random images.
2. Using the same thresholds as in Experiment #1, find all images in ToL-2 that match (are below similarity threshold).
3. Record these images to develop human intuition for what kinds of images we would filter out based on different thresholds.

The output is step #3.


## Running Experiments

Replace `$BASE_DIR` and `YOUR_ACCOUNT` with the appropriate base path and account code, then run the following code. Note that these values may also need to be replaced in other files in this directory (e.g., `imghash.slurm`).

```
uv run python main.py hash-all \
  --algorithm pdq \
  --dataset inat21/train \
  --data-root $BASE_DIR/foundation_model/inat21/raw \
  --n-workers 30 \
  --batch-size 256 \
  --slurm-acct YOUR_ACCOUNT
```

| Algorithm |  Images | Workers | Batch Size | Estimated Time |
|---|---|---|---|---|
| PDQ | 2,686,843 | 30 | 256 | 9 hours[^1] |
| PDQ | 2,686,843 | 60 | 512 | 5.5 hours |

[^1]: This is 1K hashes per hour which seems very slow. I should probably profile this to see what the slowest part is.

```
uv run python experiments.py make-hist \
  --algorithm pdq \
  --dataset inat21/train \
  --data-root $BASE_DIR/foundation_model/inat21/raw \
  --hashes hashes/inat21_train-pdq.npy \
  --n 10_000
```

Then look at `plots/inat21_train-pdq.png`.

```sh
uv run experiments.py filter-tol \
  --tol-hash-root $BASE_DIR/TreeOfLife/pdq_hash/ \
  --tol-splits eol gbif \
  --test-hashes \
    hashes/newt_test-pdq.npy \
    hashes/fishnet_test-pdq.npy \
    hashes/iwildcam_test-pdq.npy \
    hashes/insects_val-pdq.npy \
    hashes/insects2_val-pdq.npy \
    hashes/plantnet_test-pdq.npy \
    hashes/fungi_val-pdq.npy \
    hashes/plantdoc_test-pdq.npy \
    hashes/awa2_test-pdq.npy \
    hashes/herbarium19_small-validation-pdq.npy \
  --slurm-acct YOUR_ACCOUNT \
  --threshold 10 \
  --output-pf $BASE_DIR/TreeOfLife/dedup_lookup_tables/2025-05-17/exclude.parquet \
  --hours 0.5 \
  --parallelism 512
```
