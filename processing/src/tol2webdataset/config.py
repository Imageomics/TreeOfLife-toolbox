import os.path
from pathlib import Path
from typing import Any, Dict, Literal

import yaml
from attr import define

TEMPLATE_PATH = {
    "tol2webdataset": "config_templates/tol2webdataset.yaml",
}


def is_subset(template: Dict, config: Dict) -> bool:
    is_subset_result = set(template.keys()).issubset(config.keys())
    for key, value in template.items():
        if isinstance(value, dict):
            is_subset_result &= is_subset(value, config[key])
    return is_subset_result


@define
class Config:
    config_path: str
    _cfg_dict: Dict[str, str | int | bool | Dict[str, Any]]
    scripts: Dict[str, str]
    folder_structure: Dict[str, str]

    @classmethod
    def from_path(cls, path: str, config_type: Literal["tol2webdataset"]) -> "Config":
        cfg = cls.__load_config(path)

        assert cls.__check_config(cfg, config_type), "Config is not valid"

        # print(cfg)
        return cls(config_path=path,
                   cfg_dict=cfg,
                   scripts=cls.__load_scripts(cfg),
                   folder_structure=cls.__load_folder_structure(cfg))

    @staticmethod
    def __check_config(cfg: Dict[str, str | int | bool | Dict[str, Any]],
                       config_type: Literal["tol2webdataset"]) -> bool:
        config_template_path = os.path.join(Path(__file__).parent.absolute(), TEMPLATE_PATH[config_type])
        if not os.path.exists(config_template_path):
            raise FileNotFoundError(f"Config template not found, can't check it {config_template_path}")
        with open(config_template_path, "r") as f:
            config_template = yaml.full_load(f)
        return is_subset(config_template, cfg)

    @staticmethod
    def __load_config(path: str) -> Dict[str, str | int | bool | Dict[str, Any]]:
        with open(path, "r") as file:
            return yaml.full_load(file)

    @staticmethod
    def __load_scripts(cfg: Dict[str, str | int | bool | Dict[str, Any]]) -> Dict[str, str]:
        return cfg["scripts"]

    @staticmethod
    def __load_folder_structure(cfg: Dict[str, str | int | bool | Dict[str, Any]]) -> Dict[str, str]:
        file_structure: Dict[str, str] = {
            "path_to_output_folder": cfg["path_to_output_folder"],
            "path_to_source_taxa": cfg["path_to_source_taxa"],
            "path_to_image_data": cfg["path_to_image_data"],
            "path_to_image_lookup_table": cfg.get("path_to_image_lookup_table"),
        }

        file_structure.update(
            {key: os.path.join(cfg["path_to_output_folder"], value) for key, value in cfg["output_structure"].items()})

        if cfg.get("path_to_webdataset_output_folder") is not None:
            file_structure["dataset_output_folder"] = cfg.get("path_to_webdataset_output_folder")

        return file_structure

    def get_script(self, script_name: str) -> str:
        return self.scripts[script_name]

    def get_folder(self, folder_name: str) -> str:
        return self.folder_structure[folder_name]

    def get(self, key: str, default: Any = None) -> str | int | bool | Dict[str, Any]:
        return self._cfg_dict.get(key, default)

    def __getitem__(self, key: str) -> str | int | bool | Dict[str, Any]:
        return self._cfg_dict[key]