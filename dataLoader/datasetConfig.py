# FROM Python LIBRARY
from typing import List, Dict, Tuple, Union, Optional

# FROM This Project
from backbone.config import Config


class DatasetConfig:
    def __init__(
        self,
        name: str,
        origin: Optional[str] = None,
        classes: Optional[List[str]] = None,
        special = None, 
        preprocessings: Optional[List[str]] = None,
        useDatasetConfig: bool = True,
    ):

        self.name = name

        self.origin = origin
        self.classes = classes
        self.special = special
        self.preprocessings = preprocessings

        self.useDatasetConfig = useDatasetConfig

        if self.useDatasetConfig is True:
            self._getDatasetConfig()


    def _getDatasetConfig(self):

        yamlData = Config.datasetConfigDict[f"{self.name}"]

        self.origin = str(yamlData["origin"])
        self.classes = list(map(str, yamlData["classes"]))
        self.special = yamlData["special"]
        self.preprocessings = list(map(str, yamlData["preprocessings"])) if yamlData["preprocessings"] is not None else []

