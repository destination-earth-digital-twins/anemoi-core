from collections import defaultdict
from omegaconf import OmegaConf
from copy import deepcopy

class ProcessConfigs:
    def __init__(self, config):
        self.config = OmegaConf.to_container(config, resolve=True).copy()

        self.regional = self.config["dataloader"]["regional_datasets"].copy()
        
        self.training_periods = self.config["dataloader"]["training_periods"].copy()
        self.validation_periods =  self.config["dataloader"]["validation_periods"].copy()

        self.training_struct = self.config["dataloader"]["training"].copy()
        self.validation_struct = self.config["dataloader"]["validation"].copy()
        self.dataloader_config = self.config["dataloader"].copy()
        
    @property
    def modify_training(self) -> dict:

        training_dataloader_config = {}
        for region, args in self.regional.items():
            filename = deepcopy(args).pop("dataset")
            print(region)
            print(filename)
            training_struct = deepcopy(self.training_struct)
            training_struct["dataset"]["cutout"][0]["dataset"] = filename
            training_struct["start"] = self.training_periods[region]["start"]
            training_struct["end"] = self.training_periods[region]["end"]
            training_struct["dataset"]["cutout"][0].update(args)
            training_dataloader_config[region] = training_struct
            print(training_struct)
        self.dataloader_config["training"] = training_dataloader_config.copy()


    @property
    def modify_validation(self) -> dict:

        validation_dataloader_config = {}
        for region, args in self.regional.items():
            filename = deepcopy(args).pop("dataset")
            validation_struct = deepcopy(self.validation_struct)
            validation_struct["dataset"]["cutout"][0]["dataset"] = filename
            validation_struct["start"] = self.validation_periods[region]["start"]
            validation_struct["end"] = self.validation_periods[region]["end"]
            validation_struct["dataset"]["cutout"][0].update(args)
            validation_dataloader_config[region] = validation_struct
        self.dataloader_config["validation"] = validation_dataloader_config

    @property
    def combine(self) -> None:
        self.modify_training
        self.modify_validation
        self.config["dataloader"] = self.dataloader_config
       
        print("config after preprocessing ", self.dataloader_config)
        self.config = OmegaConf.create(self.config)
        return self.config


