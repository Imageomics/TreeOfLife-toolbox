import typing

import beartype
import numpy as np
import pdqhash
import pyspark.sql


def decode_and_hash_pdq(img_bytes, original_size, resized_size):
    """
    Args:
        img_bytes: raw BGR bytes from your parquet
        original_size/resized_size: [height, width] or None
        Return: 32-byte PDQ packed bits (or None if decoding fails).
    """

    if not img_bytes:
        return None

    # We decode in Python/NumPy; Spark calls this function on each row in parallel.
    np_img = np.frombuffer(img_bytes, dtype=np.uint8)

    height, width = None, None

    # Try whichever size is non-empty
    for dims in (original_size, resized_size):
        if dims and len(dims) >= 2:
            h, w = dims[:2]
            if h * w * 3 == np_img.size:
                height, width = h, w
                break

    # If we never found a valid size, skip
    if height is None or width is None:
        return None

    # Reshape from BGR to (H,W,3)
    np_img = np_img.reshape((height, width, 3))
    # Convert from BGR to RGB
    np_img = np_img[..., ::-1]

    # Compute PDQ
    try:
        hash, _ = pdqhash.compute(np_img)
        # hash is a list of 256 bits (python bools). We "pack" them into 32 bytes via np.packbits.
        bits_array = np.packbits(hash)
        return bytes(bits_array)  # Return raw bytes
    except Exception:
        return None


@beartype.beartype
def main(inputs: str | list[str], write_to: str, max_partition_bytes: str = "128MB"):
    """
    Args:
        root: Path (glob) to Parquet files containing columns ['image','original_size','resized_size']
        out: Output path (Parquet) for PDQ hashes
        max_partition_bytes: Control how large each partition can be.
    """
    # 2. Start SparkSession. This initializes a “cluster” of executors under Slurm, akin to how we might spawn multiple processes with concurrent.futures.
    spark = (
        pyspark.sql.SparkSession.builder.appName("ComputePDQ")
        # Tweak memory or partition settings as needed:
        .config("spark.sql.files.maxPartitionBytes", max_partition_bytes)
        .config("spark.executor.instances", "12")
        .config("spark.executor.cores", "12")
        .config("spark.executor.memory", "96G")
        .config("spark.driver.memory", "16G")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )

    # 3. Read the Parquet data. This is a lazily evaluated distributed DataFrame. Spark will automatically figure out how to distribute the row groups across executors.
    if isinstance(inputs, str):
        df = spark.read.parquet(inputs)
    elif isinstance(inputs, list):
        df = spark.read.parquet(*inputs)
    else:
        typing.assert_never(inputs)

    # 4. Define a Spark User-Defined Function (UDF) that wraps the decode_and_hash_pdq() logic. This tells Spark how to transform each row in parallel. We specify a return type, here BinaryType.
    pdq_udf = pyspark.sql.functions.udf(
        decode_and_hash_pdq, pyspark.sql.types.BinaryType()
    )

    # 5. Apply the UDF to produce a new column "pdq_hash".
    #    This is analogous to “map” in map-reduce: we read a row’s columns,
    #    and produce hashed bytes.
    df_hashed = df.withColumn(
        "pdq_hash",
        pdq_udf(
            pyspark.sql.functions.col("image"),
            pyspark.sql.functions.col("original_size"),
            pyspark.sql.functions.col("resized_size"),
        ),
    )

    # 6. Write the resulting DataFrame to parquet. This “action” triggers the actual job: Spark will distribute tasks, read all row groups, compute PDQ, and store results. The result will include the original columns plus the new pdq_hash column.
    df_hashed.write.mode("overwrite").parquet(write_to)

    spark.stop()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
