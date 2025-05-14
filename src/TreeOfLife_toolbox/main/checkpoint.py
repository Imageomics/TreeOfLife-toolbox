from typing import Dict

import yaml
from attr import define


@define
class Checkpoint:
    inner_checkpoint: Dict[str, bool]
    inner_checkpoint_path: str

    @classmethod
    def from_path(cls, path: str, default_structure: Dict[str, bool]) -> "Checkpoint":
        return cls(inner_checkpoint_path=path, inner_checkpoint=cls.__load_checkpoint(path, default_structure))

    @staticmethod
    def __load_checkpoint(path: str, default_structure: Dict[str, bool]) -> Dict[str, bool]:
        try:
            with open(path, "r") as file:
                checkpoint = yaml.full_load(file)
            for key, value in default_structure.items():
                if key not in checkpoint:
                    checkpoint[key] = value
            return checkpoint
        except FileNotFoundError:
            return default_structure

    def __save_checkpoint(self) -> None:
        with open(self.inner_checkpoint_path, "w") as file:
            yaml.dump(self.inner_checkpoint, file)

    def __getitem__(self, item):
        return self.inner_checkpoint[item]

    def get(self, item, default=None):
        return self.inner_checkpoint.get(item, default)

    def __setitem__(self, key, value):
        self.inner_checkpoint[key] = value
        self.__save_checkpoint()
