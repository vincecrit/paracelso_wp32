from pathlib import Path

import yaml

def path_constructor(loader: yaml.SafeLoader,
                     node: yaml.nodes.ScalarNode) -> Path:
    """Construct a path."""
    return Path(loader.construct_scalar(node)).absolute()


def get_loader() -> yaml.SafeLoader:
    loader = yaml.SafeLoader
    loader.add_constructor("!Path", path_constructor)

    return loader

def get_config_file(yml: str | Path) -> dict:
    return yaml.load(Path(yml).read_text(), Loader=get_loader())


config_file = get_config_file(Path(__file__).parent/'sarprep_config.yaml')

OUTFOLDER = config_file['OUTFOLDER']
AOI_GPKG = config_file['AOI_GPKG']
GRAPHS_WD = config_file['GRAPHS_WD']