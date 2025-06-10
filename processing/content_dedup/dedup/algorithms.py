import logging
import typing

import numpy as np
import pdqhash
from jaxtyping import UInt8

logger = logging.getLogger(__name__)

AlgorithmKey = typing.Literal["pdq", "phash"]


class Algorithm:
    def __len__(self) -> int:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def batch(
        self, imgs: list[UInt8[np.ndarray, "width height 3"]]
    ) -> UInt8[np.ndarray, "n_imgs n"]:
        raise NotImplementedError()


class Pdq(Algorithm):
    # https://github.com/faustomorales/pdqhash-python
    def __len__(self) -> int:
        return 256

    def __repr__(self) -> str:
        return "pdq"

    def batch(
        self, imgs: list[UInt8[np.ndarray, "width height 3"]]
    ) -> UInt8[np.ndarray, "n_imgs 32"]:
        hashes = np.zeros((len(imgs), 32), dtype=np.uint8)
        for i, img in enumerate(imgs):
            try:
                hash, _ = pdqhash.compute(img)
                hashes[i] = np.packbits(hash)
            except IndexError as err:
                logger.warning("Error while computing hash: %s", err)
                hashes[i] = np.zeros(32, dtype=np.uint8)
        return hashes


class Phash:
    def __len__(self) -> int:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "phash"

    def batch(
        self, imgs: list[UInt8[np.ndarray, "width height 3"]]
    ) -> UInt8[np.ndarray, "n_imgs 32"]:
        breakpoint()


def load(key: AlgorithmKey) -> Algorithm:
    if key == "pdq":
        return Pdq()
    elif key == "phash":
        return Phash()
    else:
        typing.assert_never(key)
