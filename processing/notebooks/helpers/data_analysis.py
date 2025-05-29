from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.sql.functions import col, desc, when


def init_spark(session_name = "GBIF EDA") ->  SparkSession:
    # spark = SparkSession.builder \
    #     .appName("GBIF EDA") \
    #     .getOrCreate()
    # spark = SparkSession.builder \
    #     .appName("GBIF EDA") \
    #     .config("spark.executor.instances", "36") \
    #     .config("spark.executor.cores", "5") \
    #     .config("spark.executor.memory", "20g") \
    #     .config("spark.sql.shuffle.partitions", "180") \
    #     .config("spark.task.cpus", "1") \
    #     .getOrCreate()
    spark = SparkSession.builder \
        .appName(session_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.instances", "40") \
        .config("spark.executor.cores", "10") \
        .config("spark.executor.memory", "40g") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.parquet.columnarReaderBatchSize", "512") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "10g") \
        .config("spark.sql.sources.bucketing.enabled", "false") \
        .getOrCreate()
    return spark

def create_freq(df: DataFrame, col_name: list) -> DataFrame:
    if isinstance(col_name, str):
        col_name = [col_name]
        
    freq_tbl = (
        df
        .groupBy(*col_name)
        .count()
        .orderBy(desc("count"))
        .withColumn(
            "bucket",
            when(col("count") <= 10, "1-10")
            .when(col("count") <= 50, "11-50")
            .when(col("count") <= 100, "51-100")
            .when(col("count") <= 500, "101-500")
            .when(col("count") <= 1_000, "501-1k")
            .when(col("count") <= 5_000, "1k-5k")
            .when(col("count") <= 10_000, "5k-10k")
            .when(col("count") <= 50_000, "10k-50k")
            .when(col("count") <= 100_000, "50k-100k")
            .when(col("count") <= 500_000, "100k-500k")
            .when(col("count") <= 1_000_000, "500k-1m")
            .when(col("count") <= 5_000_000, "1m-5m")
            .when(col("count") <= 10_000_000, "5m-10m")
            .otherwise("10m+")
        )
    )
    return freq_tbl


def view_freq(df: DataFrame, col_name: list, num_rows: int = 20, truncate: bool = False) -> None:
    
    create_freq(df, col_name).show(num_rows, truncate=truncate)


def check_sparsity(df: DataFrame) -> DataFrame:
    # Count the total number of rows in the DataFrame once
    total_rows = df.count()
    
    # Create a list of column-wise expressions to compute null and zero counts
    sparsity_exprs = [
        (F.sum(F.when((F.col(col).isNull()) | (F.col(col) == 0), 1).otherwise(0)) / total_rows).alias(col)
        for col in df.columns
    ]
    
    # Perform the aggregation in a single pass
    sparsity_row = df.agg(*sparsity_exprs).collect()[0]
    
    # Convert the result into a DataFrame
    sparsity_data = [(col, sparsity_row[col]) for col in df.columns]
    schema = T.StructType([
        T.StructField("column_name", T.StringType(), True),
        T.StructField("sparsity", T.DoubleType(), True)
    ])
    
    sparsity_df = df.sparkSession.createDataFrame(sparsity_data, schema)
    
    return sparsity_df
