from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from typing import List, Sequence, Optional, Dict, Any

import numpy as np
from attrs import define
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType


def load_dataframe(
        spark: SparkSession, input_path: str, scheme: Optional[StructType | str] = None
) -> DataFrame:
    file_extension = input_path.split(".")[-1].lower()

    def infer_delimiter(_first_line):
        if "\t" in _first_line:
            return "\t"
        elif "," in _first_line:
            return ","
        elif " " in _first_line:
            return " "
        elif "|" in _first_line:
            return "|"
        elif ";" in _first_line:
            return ";"
        else:
            return None

    if file_extension in ["csv", "tsv", "txt"]:
        if file_extension == "csv":
            sep = ","
        elif file_extension == "tsv":
            sep = "\t"
        elif file_extension == "txt":
            with open(input_path, "r") as file:
                first_line = file.readline()
                sep = infer_delimiter(first_line)
            if sep is None:
                raise ValueError(f"Could not infer delimiter for file {input_path}")
        df = spark.read.csv(input_path, sep=sep, header=True, schema=scheme)
    else:
        try:
            df = spark.read.load(input_path, scheme=scheme)
        except Exception as e:
            raise FileNotFoundError(f"File not supported: {e}")

    return df


def ensure_created(list_of_path: Sequence[str]) -> None:
    for path in list_of_path:
        os.makedirs(path, exist_ok=True)


def truncate_paths(paths: Sequence[str]) -> None:
    for path in paths:
        is_dir = "." not in path.split("/")[-1]
        if is_dir:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        else:
            open(path, "w").close()


def get_id(output: bytes) -> int:
    return int(output.decode().strip().split(" ")[-1])


def init_logger(
        logger_name: str, output_path: str = None, logging_level: str = "INFO"
) -> logging.Logger:
    logging.basicConfig(
        filename=output_path,
        level=logging.getLevelName(logging_level),
        format="%(asctime)s - %(levelname)s - %(process)d - %(message)s",
    )
    return logging.getLogger(logger_name)


def submit_job(submitter_script: str, script: str, *args) -> int:
    output = subprocess.check_output(
        f"{submitter_script} {script} {' '.join(args)}", shell=True
    )
    idx = get_id(output)
    return idx


def preprocess_dep_ids(ids: List[int | None]) -> List[str]:
    return [str(_id) for _id in ids if _id is not None]


_NOT_PROVIDED = "Not provided"


@define
class DownloadedImage:
    retry_count: int
    error_code: int
    error_msg: str

    unique_name: str
    source_id: int
    identifier: str
    is_license_full: bool
    license: str
    source: str
    title: str

    hashsum_original: str = ""
    hashsum_resized: str = ""
    # image: np.ndarray = np.ndarray(0)
    image: bytes = bytes()
    original_size: np.ndarray[np.uint32] = np.ndarray([0, 0], dtype=np.uint32)
    resized_size: np.ndarray[np.uint32] = np.ndarray([0, 0], dtype=np.uint32)

    start_time: float = 0
    end_time: float = 0

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> DownloadedImage:
        if "EOL content ID" in row.keys() and "EOL page ID" in row.keys():
            source_id = row["EOL content ID"] + "_" + row["EOL page ID"]
        else:
            source_id = "None"

        return cls(
            retry_count=0,
            error_code=0,
            error_msg="",
            unique_name=row.get("uuid", uuid.uuid4().hex),
            source_id=row.get("source_id", source_id),
            identifier=row.get("identifier", ""),
            is_license_full=all(
                [
                    row.get("license", None),
                    row.get("source", None),
                    row.get("title", None),
                ]
            ),
            license=row.get("license", _NOT_PROVIDED) or _NOT_PROVIDED,
            source=row.get("source", _NOT_PROVIDED) or _NOT_PROVIDED,
            title=row.get("title", _NOT_PROVIDED) or _NOT_PROVIDED,
        )


def init_downloaded_image_entry(
        image_entry: np.ndarray, row: Dict[str, Any]
) -> np.ndarray:
    image_entry["is_downloaded"] = False
    image_entry["retry_count"] = 0
    image_entry["error_code"] = 0
    image_entry["error_msg"] = ""
    image_entry["uuid"] = row.get("UUID", uuid.uuid4().hex)
    image_entry["source_id"] = row.get("source_id", 0)
    image_entry["identifier"] = row.get("identifier", "")
    image_entry["is_license_full"] = all(
        [row.get("license", None), row.get("source", None), row.get("title", None)]
    )
    image_entry["license"] = row.get("license", _NOT_PROVIDED) or _NOT_PROVIDED
    image_entry["source"] = row.get("source", _NOT_PROVIDED) or _NOT_PROVIDED
    image_entry["title"] = row.get("title", _NOT_PROVIDED) or _NOT_PROVIDED

    return image_entry


@define
class SuccessEntry:
    uuid: str
    source_id: int
    identifier: str
    is_license_full: bool
    license: str
    source: str
    title: str
    hashsum_original: str
    hashsum_resized: str
    original_size: np.ndarray[np.uint32]
    resized_size: np.ndarray[np.uint32]
    image: bytes

    def __success_dtype(self, img_size: int):
        return np.dtype(
            [
                ("uuid", "S32"),
                ("source_id", "S32"),
                ("identifier", "S256"),
                ("is_license_full", "bool"),
                ("license", "S256"),
                ("source", "S256"),
                ("title", "S256"),
                ("original_size", "(2,)u4"),
                ("resized_size", "(2,)u4"),
                ("hashsum_original", "S32"),
                ("hashsum_resized", "S32"),
                ("image", f"({img_size},{img_size},3)uint8"),
            ]
        )

    @staticmethod
    def get_success_spark_scheme():
        from pyspark.sql.types import StructType
        from pyspark.sql.types import StringType
        from pyspark.sql.types import LongType
        from pyspark.sql.types import StructField
        from pyspark.sql.types import BooleanType
        from pyspark.sql.types import ArrayType
        from pyspark.sql.types import BinaryType

        return StructType(
            [
                StructField("uuid", StringType(), False),
                StructField("source_id", StringType(), False),
                StructField("identifier", StringType(), False),
                StructField("is_license_full", BooleanType(), False),
                StructField("license", StringType(), True),
                StructField("source", StringType(), True),
                StructField("title", StringType(), True),
                StructField("original_size", ArrayType(LongType(), False), False),
                StructField("resized_size", ArrayType(LongType(), False), False),
                StructField("hashsum_original", StringType(), False),
                StructField("hashsum_resized", StringType(), False),
                StructField("image", BinaryType(), False),
            ]
        )

    @classmethod
    def from_downloaded(cls, downloaded: DownloadedImage) -> SuccessEntry:
        return cls(
            uuid=downloaded.unique_name,
            source_id=downloaded.source_id,
            identifier=downloaded.identifier,
            is_license_full=downloaded.is_license_full,
            license=downloaded.license,
            source=downloaded.source,
            title=downloaded.title,
            hashsum_original=downloaded.hashsum_original,
            hashsum_resized=downloaded.hashsum_resized,
            original_size=downloaded.original_size,
            resized_size=downloaded.resized_size,
            image=downloaded.image,
        )

    @staticmethod
    def to_list_download(downloaded: DownloadedImage) -> List:
        return [
            downloaded.unique_name,
            downloaded.source_id,
            downloaded.identifier,
            downloaded.is_license_full,
            downloaded.license,
            downloaded.source,
            downloaded.title,
            downloaded.original_size,
            downloaded.resized_size,
            downloaded.hashsum_original,
            downloaded.hashsum_resized,
            downloaded.image,
        ]

    @staticmethod
    def get_names() -> List[str]:
        return [
            "uuid",
            "source_id",
            "identifier",
            "is_license_full",
            "license",
            "source",
            "title",
            "original_size",
            "resized_size",
            "hashsum_original",
            "hashsum_resized",
            "image",
        ]

    def to_list(self) -> List:
        return [
            self.uuid,
            self.source_id,
            self.identifier,
            self.is_license_full,
            self.license,
            self.source,
            self.title,
            self.original_size,
            self.resized_size,
            self.hashsum_original,
            self.hashsum_resized,
            self.image,
        ]

    def to_np(self) -> np.ndarray:
        np_structure = np.array(
            [
                (
                    self.uuid,
                    self.source_id,
                    self.identifier,
                    self.is_license_full,
                    self.license,
                    self.source,
                    self.title,
                    self.original_size,
                    self.resized_size,
                    self.hashsum_original,
                    self.hashsum_resized,
                    self.image,
                )
            ],
            dtype=self.__success_dtype(np.max(self.resized_size)),
        )

        return np_structure
