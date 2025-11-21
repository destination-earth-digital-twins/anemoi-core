from copy import deepcopy
from collections import defaultdict

import hydra
from omegaconf import OmegaConf, DictConfig

from anemoi.training.schemas.base_schema import convert_to_omegaconf


class ProcessConfigs:
    SENTINEL = object()
    TEMPORARY = defaultdict(dict)

    def __init__(
        self,
        base_config: DictConfig,
        hectometric: bool = False,
    ) -> None:
        """
        Initialize the ProcessConfigs with the base configuration.

        args:
            base_config (DictConfig): The base configuration object.
            hectometric (bool): Flag indicating if hectometric processing is needed.
        returns:
            None
        """
        OmegaConf.resolve(base_config)
        self.config = OmegaConf.to_container(base_config, resolve=True)

        self.struct = self.config["dataloader"]
        if hectometric:
            self.struct_train = getattr(
                self.config["dataloader"], "hectometric_dataset_training", None
            )
            self.struct_val = getattr(
                self.config["dataloader"], "hectometric_dataset_validation", None
            )
            assert (
                self.struct_train is not None
            ), f"Hectometric run enabled, hectometric training dataset path is not provided."
            assert (
                self.struct_train is not None
            ), f"Hectometric run enabled, hectometric validation dataset path is not provided."
        else:
            self.regional = self.config["dataloader"]["regional_datasets"]

    def _findcutoutnulls(self, cutout, replacement: dict) -> dict:
        """
        Recursively search through the cutout structure
        to find any dicts where "dataset" and other keys
        is explicitly None, and replace that dict with
        the provided replacement dict.

        args:
            cutout (dict): The cutout configuration structure.
            replacement (dict): The replacement dictionary to use.
        returns:
            dict: The modified cutout structure with replacements made.
        """

        def recurse(obj):
            if isinstance(obj, dict):
                # If this dict explicitly has dataset=None
                if obj.get("dataset", self.SENTINEL) is None:
                    # Replace the whole dict content with a deep copy of the replacement
                    obj.clear()
                    obj.update(replacement.copy())
                else:
                    # Otherwise, keep traversing deeper
                    for k, v in list(obj.items()):
                        recurse(v)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if (
                        isinstance(item, dict)
                        and item.get("dataset", self.SENTINEL) is None
                    ):
                        # Replace the entire element if it has dataset=None
                        obj[i] = replacement.copy()
                    else:
                        recurse(item)

        recurse(cutout)
        return cutout

    def _inject_date(self, struct, start, end):
        """
        Inject values into the config where None exists,
        based on a dictionary mapping.
        Example: {"cutout[0].dataset": "some_path"}
        """
        assert (
            start is not None and end is not None
        ), "Start and end dates must be provided."
        assert start <= end, "Start date must be less than or equal to end date."

        struct["start"] = start
        struct["end"] = end

        return struct

    @property
    def process(self):
        """
        Process the configurations to replace
        cutout nulls with regional datasets
        and inject dates

        args:
            None
        returns:
            None
        """

        if self.hectometric:
            for name, phase in [
                (self.struct_train, "training"),
                (self.struct_val, "validation"),
            ]:
                with open(name, "r") as f:
                    for lines in f:
                        filename = lines
                        key = filename.split(".")[0]

                        splitted = lines.split("_")

                        start, end = splitted[1:3]
                        start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
                        end = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
                        self.TEMPORARY[phase][key] = self._findcutoutnulls(
                            deepcopy(self.struct[phase]),
                            replacement={"dataset": filename},
                        )
                        self.TEMPORARY[phase][key] = self._inject_date(
                            self.TEMPORARY[phase][key],
                            start=start,
                            end=end,
                        )
        else:
            for phase in ["training", "validation"]:
                for region, args in self.regional.items():
                    self.TEMPORARY[phase][region] = self._findcutoutnulls(
                        deepcopy(self.struct[phase]), replacement=args
                    )

                    # print(self.TEMPORARY[phase][region])
                    self.TEMPORARY[phase][region] = self._inject_date(
                        self.TEMPORARY[phase][region],
                        start=self.struct[f"{phase}_periods"][region]["start"],
                        end=self.struct[f"{phase}_periods"][region]["end"],
                    )

    def update(self):
        """
        Update the base configuration with the processed temporary structures
        containing information of each regional domain.

        args:
            None
        returns:
            DictConfig: The updated configuration object.

        """
        for phase, structs in self.TEMPORARY.items():
            self.config["dataloader"][phase] = structs
        return OmegaConf.create(self.config)
