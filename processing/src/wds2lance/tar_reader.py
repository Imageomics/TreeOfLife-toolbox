"""Read samples out of a webdataset tar shard.

A sample is a run of consecutive files that share a key, the file name up to its first ".": the files
abc.jpg, abc.sci.txt, ... form the sample "abc" with members "jpg", "sci.txt", ...
"""
import tarfile
from typing import Iterator


def read_samples(tar_path: str) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Yield (key, {member: bytes}) for each sample in tar order, e.g. ("abc", {"jpg": b"...", "sci.txt": b"..."})."""
    key, sample = None, {}
    with tarfile.open(tar_path, mode="r|") as tar:  # stream the tar front to back, without seeking
        for member in tar:
            if not member.isreg():  # skip directories and links
                continue
            member_key, suffix = member.name.split(".", 1)
            if member_key != key:  # a new key, so the previous sample is complete
                if sample:
                    yield key, sample
                key, sample = member_key, {}
            sample[suffix] = tar.extractfile(member).read()
            tar.members = []  # stream mode otherwise keeps every file header in memory
    if sample:
        yield key, sample
