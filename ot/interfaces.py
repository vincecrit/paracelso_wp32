import json
from abc import ABC
from pathlib import Path


class OTAlgorithm(ABC):

    R, G, B = 0.21250, 0.71540, 0.07210

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

    @staticmethod
    def show_name(self): return self.__class__.__name__

    def __call__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    # @classmethod # magari inserirla in main.py
    # def run(cls, **kwargs) -> None:
    #     """
    #     args = get_parser().parse_args()
    #     algorithm = get_algorithm(args.algname)
    #     kwargs = vars(args) # superflua probabilemente

    #     OTMethod = algorithm.from_dict(kwargs)

    #     # sostituire cls con OTMethod
    #     """

    #     OTMethod = cls.from_dict(kwargs)

    #     target_coreg = Path(kwargs['target']).parent / \
    #         (Path(kwargs['target']).stem+'_coreg.tif')

    #     basic_pixel_coregistration(
    #         kwargs['target'],
    #         kwargs['reference'],
    #         target_coreg)

    #     with rasterio.open(kwargs['reference']) as refimg:
    #         with rasterio.open(target_coreg) as tarimg:
    #             reference_array = refimg.read(
    #                 kwargs['band']).astype(np.float32)
    #             target_array = tarimg.read(kwargs['band']).astype(np.float32)
    #             target_mask = np.logical_not(
    #                 tarimg.dataset_mask().astype(bool))
    #             target_metadata = tarimg.meta.copy()

    #     if kwargs['rgb2gray']:
    #         if kwargs['band'] is not None:
    #             print("Se si vuole convertire in scala di grigi, non è possibile specificare una singola banda")
    #             exit(1)

    #         reference_array = np.dot(reference_array[..., :3], [cls.R, cls.G, cls.B])
    #         target_array = np.dot(target_array[..., :3], [cls.R, cls.G, cls.B])
        
    #     elif all([kwargs['rgb2gray'], kwargs['band'] is not None]):
    #         print("Specificare almeno uno tra 'rgb2gray' e 'band'")
    #         exit(1)

    #     target_metadata['count'] = 1

    #     resdispl = OTMethod(target_metadata, reference_array, target_array)

    #     target_metadata['dtype'] = 'float32'
    #     resdispl_ma = np.ma.masked_array(resdispl, target_mask)
    #     print(np.min(resdispl_ma))
    #     print(np.max(resdispl_ma))

    #     with rasterio.open(kwargs["output"], "w", **target_metadata) as dst:
    #         for iband in target_metadata["count"]:
    #             dst.write(iband, resdispl_ma[iband])
