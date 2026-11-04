import tarfile
from typing import Iterator


def read_samples(tar_path: str) -> Iterator[tuple[str, dict[str, bytes]]]:
    key, sample = None, {}
    with tarfile.open(tar_path, mode="r|") as tar:  # stream front to back, without seeking
        for member in tar:
            if not member.isreg():
                continue
            member_key, suffix = member.name.split(".", 1)
            if member_key != key:
                if sample:
                    yield key, sample
                key, sample = member_key, {}
            sample[suffix] = tar.extractfile(member).read()
            tar.members = []  # stream mode otherwise keeps every file header in memory
    if sample:
        yield key, sample
