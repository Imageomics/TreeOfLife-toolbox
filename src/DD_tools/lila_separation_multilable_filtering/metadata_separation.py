from pyspark.sql import SparkSession

base_path = "/fs/scratch/PAS2136/gbif/processed/lilabc/merged_data/servers_batched"
filter_path = "/users/PAS2119/andreykopanev/gbif/data/lila_separation_table/part-00000-6e425202-ecec-426d-9631-f2f52fd45c51-c000.csv"
save_path = "/fs/scratch/PAS2136/gbif/processed/lilabc/separated_multilabel_data/servers_batched"

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Multimedia prep").getOrCreate()
    spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")

    metadata_df = spark.read.parquet(base_path).drop("partition_id")
    filter_df = spark.read.csv(filter_path, header=True).select("uuid", "partition_id")

    df = metadata_df.join(filter_df, on="uuid", how="inner")

    (df
     .repartition("server_name", "partition_id")
     .write
     .partitionBy("server_name", "partition_id")
     .mode("overwrite")
     .format("parquet")
     .save(save_path))

    spark.stop()
