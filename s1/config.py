from pathlib import Path

import yaml


def path_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> Path:
    """Construct a path."""
    return Path(loader.construct_scalar(node))  # .absolute()


def get_loader() -> yaml.SafeLoader:
    loader = yaml.SafeLoader
    loader.add_constructor("!Path", path_constructor)

    return loader

def get_config_file(yml: str | Path) -> dict:
    return yaml.load(Path(yml).read_text(), Loader=get_loader())

OUTFOLDER = get_config_file(Path(__file__).parent/'sarprep_config.yaml')['OUTFOLDER']
AOI_GPKG = get_config_file(Path(__file__).parent/'sarprep_config.yaml')['AOI_GPKG']
GRAPHS_WD = get_config_file(Path(__file__).parent/'sarprep_config.yaml')['GRAPHS_WD']
