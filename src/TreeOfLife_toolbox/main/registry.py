from typing import Dict, Type, Optional

from DD_tools.main.config import Config
from DD_tools.main.utils import init_logger


class ToolsRegistryBase(type):
    TOOLS_REGISTRY: Dict[str, Dict[str, Type["ToolsBase"]]] = {}

    @classmethod
    def get(cls, name):
        return cls.TOOLS_REGISTRY.get(name.lower())

    @classmethod
    def register(cls, filter_family: str, filter_name: str):
        def wrapper(model_cls):
            assert issubclass(model_cls, ToolsBase)
            assert (
                    filter_name not in cls.TOOLS_REGISTRY.keys()
                    or filter_family not in cls.TOOLS_REGISTRY[filter_name].keys()
            ), ValueError(
                f"tool with the name {filter_name} already have family {filter_family}"
            )

            if filter_name not in cls.TOOLS_REGISTRY.keys():
                cls.TOOLS_REGISTRY[filter_name] = dict()

            cls.TOOLS_REGISTRY[filter_name][filter_family] = model_cls
            return model_cls

        return wrapper

    def __contains__(self, item):
        return item in self.TOOLS_REGISTRY

    def __iter__(self):
        return iter(self.TOOLS_REGISTRY)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.TOOLS_REGISTRY})"

    __str__ = __repr__


class ToolsBase(metaclass=ToolsRegistryBase):
    # noinspection PyTypeChecker
    def __init__(self, cfg: Config):
        self.config = cfg

        self.filter_name: Optional[str] = None
        self.filter_family: Optional[str] = None

        self.logger = init_logger(__name__)

        self.urls_path = self.config.get_folder("urls_folder")
        self.downloaded_images_path = self.config.get_folder("images_folder")
        self.tools_path = self.config.get_folder("tools_folder")
        self.total_workers = (
                self.config["tools_parameters"]["max_nodes"]
                * self.config["tools_parameters"]["workers_per_node"]
        )

    def run(self):
        raise NotImplementedError()
