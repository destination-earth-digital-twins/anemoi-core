# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Callable

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
#from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.data.multidomain_dataset import NativeMultiGridDataset

from anemoi.training.data.dataset import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.training.data.grid_indices import BaseGridIndices


class AnemoiMultiDomainDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, graph_data: HeteroData) -> None:
        """Initialize Anemoi Datasets data module.

        Parameters
        ----------
        config : DictConfig
            Job configuration

        """
        super().__init__()

        self.config = config
        self.graph_data = graph_data

        # Set the maximum rollout to be expected
        self.rollout = (
            self.config.training.rollout.max
            if self.config.training.rollout.epoch_increment > 0
            else self.config.training.rollout.start
        )
        
        resolved_training_conf = OmegaConf.to_container(self.config.dataloader.training, resolve = True)
        for dataset_label in resolved_training_conf:
            # Set the training end date if not specified
            if resolved_training_conf[dataset_label]["end"] is None:
                LOGGER.info(
                   f"No end date specified for {dataset_label} training data, setting default before validation start date %s.",
                    resolved_training_conf[dataset_label]["start"] - 1,
                )
                self.config.dataloader.training.end = resolved_training_conf[dataset_label]["start"] - 1

        if not self.config.dataloader.get("pin_memory", True):
            LOGGER.info("Data loader memory pinning disabled.")
    
    @cached_property
    def statistics(self) -> dict:
        return self.ds_train.statistics

    @cached_property
    def metadata(self) -> dict:
        return self.ds_train.metadata

    @cached_property
    def supporting_arrays(self) -> dict:
        return self.ds_train.supporting_arrays | self.grid_indices[list(self.grid_indices.keys())[0]].supporting_arrays

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)

    @cached_property
    def grid_indices(self) -> type[BaseGridIndices]:
        """

            grid indices for multi domain 
        """
        reader_group_size = self.config.dataloader.get("read_group_size", self.config.hardware.num_gpus_per_model)
        grid_indices_dict = {}
        for graph_label, graph in self.graph_data.items():
            grid_indices = instantiate(self.config.dataloader.grid_indices, reader_group_size=reader_group_size)
            grid_indices.setup(graph)
            grid_indices_dict[graph_label] = grid_indices
        return grid_indices_dict

    @cached_property
    def timeincrement(self) -> int:
        """Determine the step size relative to the data frequency."""
        try:
            frequency = frequency_to_seconds(self.config.data.frequency)
        except ValueError as e:
            msg = f"Error in data frequency, {self.config.data.frequency}"
            raise ValueError(msg) from e

        try:
            timestep = frequency_to_seconds(self.config.data.timestep)
        except ValueError as e:
            msg = f"Error in timestep, {self.config.data.timestep}"
            raise ValueError(msg) from e

        assert timestep % frequency == 0, (
            f"Timestep ({self.config.data.timestep} == {timestep}) isn't a "
            f"multiple of data frequency ({self.config.data.frequency} == {frequency})."
        )
        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            timestep // frequency,
            frequency,
            timestep,
        )
        return timestep // frequency

    @cached_property
    def ds_train(self) -> NativeMultiGridDataset:
        return self._get_dataset(OmegaConf.to_container(self.config.dataloader.training, resolve=True),
            label="train",
        )

    @cached_property
    def ds_valid(self) -> NativeMultiGridDataset:
        r = max(self.rollout, self.config.dataloader.get("validation_rollout", 1))
        resolved_dataloader_conf = OmegaConf.to_container(self.config.dataloader, resolve = True)
        for dataset_label in resolved_dataloader_conf["training"]:
            if not resolved_dataloader_conf["training"][dataset_label]["end"] < resolved_dataloader_conf["validation"][dataset_label]["start"]:
                LOGGER.warning(
                    "Training end date %s is not before validation start date %s for dataset %s.",
                    self.config.dataloader.training.end,
                    self.config.dataloader.validation.start,
                    dataset_label,
                )
        return self._get_dataset(
            #open_dataset(
            OmegaConf.to_container(self.config.dataloader.validation, resolve=True),
            shuffle=False,
            rollout=r,
            label="validation",
        )

    @cached_property
    def ds_test(self) -> NativeMultiGridDataset:
        assert self.config.dataloader.training.end < self.config.dataloader.test.start, (
            f"Training end date {self.config.dataloader.training.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        assert self.config.dataloader.validation.end < self.config.dataloader.test.start, (
            f"Validation end date {self.config.dataloader.validation.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.test, resolve=True)),
            shuffle=False,
            label="test",
        )

    def _get_dataset(
        self,
        data_reader: dict[Callable],
        shuffle: bool = True,
        rollout: int = 1,
        label: str = "generic",
    ) -> NativeMultiGridDataset:

        r = max(rollout, self.rollout)

        # Compute effective batch size
        effective_bs = (
            self.config.dataloader.batch_size["training"]
            * self.config.hardware.num_gpus_per_node
            * self.config.hardware.num_nodes
            // self.config.hardware.num_gpus_per_model
        )

        data_readers = {}
        #dataset = data_reader.pop("dataset")
        for dataset_label, dataset_config in data_reader.items():
            data_readers[dataset_label] = open_dataset(dataset_config)

        print("MULTIDOMAIN WEIGHTING ", self.config.dataloader.dataset_weights)

        return NativeMultiGridDataset(
            data_readers=data_readers,
            rollout=r,
            multistep=self.config.training.multistep_input,
            timeincrement=self.timeincrement,
            shuffle=shuffle,
            grid_indices=self.grid_indices,
            dataset_weights=self.config.dataloader.dataset_weights,
            label=label,
            effective_bs=effective_bs,
        )

    def _get_dataloader(self, ds: NativeMultiGridDataset, stage: str) -> DataLoader:
        assert stage in {"training", "validation", "test"}
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            # number of worker processes
            num_workers=self.config.dataloader.num_workers[stage],
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=self.config.dataloader.get("pin_memory", True),
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")
