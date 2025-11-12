# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable

import torch
from omegaconf import DictConfig

from anemoi.datasets.data import open_dataset
from anemoi.training.data.dataset import EnsNativeGridDataset
from anemoi.training.utils.worker_init import worker_init_func

from .singledatamodule import AnemoiDatasetsDataModule

LOGGER = logging.getLogger(__name__)


class AnemoiEnsDatasetsDataModule(AnemoiDatasetsDataModule):
    """
    Anemoi Ensemble data module for PyTorch Lightning.
    Inherits from parent class AnemoiDatasetsDataModule.
    
    Constructor and its method are inherited from AnemoiDatasetsDataModule,
    required parameters:
    
    Parameters
        ----------
        config : BaseSchema
            Job configuration
        graph_data: HeteroData
            graph and its information

    """

    @staticmethod
    def collate_fn(batch) -> list[list[torch.Tensor],str]:
        """
        Collate function for multi-domain datasets.
        
        Args:
            batch: List of tuples -> [((tensor,), domain), ((tensor,), domain), ...]
        
        Returns:
            tuple(tuple(tensor), list[str])
            -> ((batched_tensor,), domains)
        """
        # Separate tensors and domain labels
        samples, domains = zip(*batch)  # unzip into ((tensor,), (tensor,), ...), (domain, domain, ...)

        assert all(domains[0] == d for d in domains), "All samples in the batch must belong to the same domain."

        # Since each sample is a tuple (usually (tensor,)), unpack the inner tensors
        tensors = [s for s in samples]
        
        # Stack tensors into a batch
        batched_tensor = torch.stack(tensors, dim=0)

        # Return ((batched_tensor,), list_of_domains)
        return list(((batched_tensor,), domains[0]))

    def _get_dataset(
        self,
        data_reader: Callable | DictConfig[str, dict],
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
    ) -> EnsNativeGridDataset:

        if isinstance(data_reader, DictConfig) and self.dynamic_mode:
            data_reader = {
                domain : open_dataset(dataset_config) for domain, dataset_config in data_reader.items()
                }
            data_reader = {
                domain : self.add_trajectory_ids(reader) for domain, reader in data_reader.items()
            }
        else:
            data_reader = self.add_trajectory_ids(data_reader)  # NOTE: Functionality to be moved to anemoi datasets

        return EnsNativeGridDataset(
            data_reader=data_reader,
            relative_date_indices=self.relative_date_indices(val_rollout),
            timestep=self.config.data.timestep,
            shuffle=shuffle,
            grid_indices=self.grid_indices,
            label=label,
            ens_members_per_device=self.config.training.ensemble_size_per_device,
            num_gpus_per_ens=self.config.hardware.num_gpus_per_ensemble,
            num_gpus_per_model=self.config.hardware.num_gpus_per_model,
            dynamic_mode=self.dynamic_mode,
            dataset_weights=self.dataset_weights,
        )
