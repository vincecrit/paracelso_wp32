import json
from abc import ABC
from pathlib import Path


class OTAlgorithm(ABC):

    def from_dict(__d: dict): ...

    @staticmethod
    def from_JSON(__json: Path | str):
        __d = json.loads(Path(__json).read_text())

        return OTAlgorithm.from_dict(__d)

    @staticmethod
    def from_YAML(__yaml: Path | str):
        import yaml

        __d = yaml.safe_load(Path(__yaml).read_text())

        return OTAlgorithm.from_dict(__d)

    def toJSON(self):

        try:
            parms = self.__dict__
            parms['flags'] = self.flags.value
        except AttributeError:
            parms = self.__dict__

        return json.dumps(parms, indent=4)

    def __call__(self, *args) -> None:
        print("Running: ", self.__class__.__name__)
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None: ...
