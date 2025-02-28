from collections import defaultdict
from omegaconf import OmegaConf


class ProcessConfigs:
    def __init__(self, config):
        self.config = OmegaConf.to_container(config, resolve=True)

        self.regional = self.config["dataloader"]["regional_datasets"]
        
        self.training_periods = self.config["dataloader"]["training_periods"]
        self.validation_periods =  self.config["dataloader"]["validation_periods"]

        self.training_struct = self.config["dataloader"]["training"]
        self.validation_struct = self.config["dataloader"]["validation"]
        
        self.new_dataloader_config = {}
    @property
    def modify_training(self) -> dict:
        # key : dataset
        # value: switch with none to path of the dataset
        #for key, value in self.config["dataloader"].items():
        #    print(key, value)

        training_dataloader_config = {}

        for region, filename in self.regional.items():
            print(region, filename)
            self.training_struct["dataset"]["cutout"][0]["dataset"] = filename
            self.training_struct["start"] = self.training_periods[region]["start"]
            self.training_struct["end"] = self.training_periods[region]["end"]
            print(self.training_struct)
            training_dataloader_config[region] = self.training_struct
        
        self.new_dataloader_config["training"] = training_dataloader_config

    @property
    def modify_validation(self) -> dict:
        validation_dataloader_config = {}


        for region, filename in self.regional.items():
            self.validation_struct["dataset"]["cutout"][0]["dataset"] = filename
            self.validation_struct["start"] = self.validation_periods[region]["start"]
            self.validation_struct["end"] = self.validation_periods[region]["end"]

            validation_dataloader_config[region] = self.validation_struct
        self.new_dataloader_config["validation"] = validation_dataloader_config

    @property
    def combine(self) -> None:
        self.modify_training
        self.modify_validation

        self.config["dataloader"]["training"] = self.new_dataloader_config["training"]
        self.config["dataloader"]["validation"] = self.new_dataloader_config["validation"]
        print("####################################")
        print(self.config)
        self.config = OmegaConf.create(self.config)
        print(self.config.dataloader.training)


