from typing import Iterator

from webdataset.tariterators import group_by_keys, tar_file_expander


def read_samples(tar_path: str) -> Iterator[tuple[str, dict[str, bytes]]]:
    with open(tar_path, "rb") as stream:
        files = tar_file_expander([{"url": tar_path, "stream": stream}])
        try:
            # lcase=False: webdataset otherwise lowercases taxonTag.txt
            for sample in group_by_keys(files, lcase=False):
                key = sample.pop("__key__")
                yield key, {name: data for name, data in sample.items()
                            if not name.startswith("__")}
        except Exception as error:
            # webdataset adds the open file to the error, which a worker process cannot send back
            raise ValueError(f"{tar_path} cannot be read: {error.args[0]}") from error
