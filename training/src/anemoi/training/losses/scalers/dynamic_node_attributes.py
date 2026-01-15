# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch_geometric.data import HeteroData

from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.masks import BaseMask
from anemoi.training.utils.masks import NoOutputMask

LOGGER = logging.getLogger(__name__)


class DynamicGraphNodeAttributeScaler(BaseUpdatingScaler):
    """Class for extracting scalers from node attributes."""

    scale_dims: TensorDim = TensorDim.GRID

    def __init__(
        self,
        nodes_name: str,
        nodes_attribute_name: str | None = None,
        inverse: bool = False,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        nodes_name : str
            Name of the nodes in the graph.
        nodes_attribute_name : str | None, optional
            Name of the node attribute to use for scaling, by default None
        output_mask : type[BaseMask], optional
            Whether to apply output mask to the scaling, by default False
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(norm=norm)
        del kwargs
        # self.output_mask = output_mask if output_mask is not None else NoOutputMask()
        # self.nodes = graph_data[nodes_name]
        self.nodes_name = nodes_name
        self.nodes_attribute_name = nodes_attribute_name
        self.inverse = inverse

    def get_scaling_values(self) -> torch.Tensor:
        scaler_values = self.nodes[self.nodes_attribute_name].squeeze()
        scaler_values = ~scaler_values if self.inverse else scaler_values
        return self.output_mask.apply(scaler_values, dim=0, fill_value=0.0)

    def on_batch_start(self, model):
        self.output_mask = model.model.model.output_mask if model.model.model.output_mask is not None else NoOutputMask()
        self.nodes = model.model.model.graph[self.nodes_name]
        scaling_values = self.get_scaling_values()
        return scaling_values

class DynamicReweightedGraphNodeAttributeScaler(DynamicGraphNodeAttributeScaler):
    """Class for extracting and reweighting node attributes.

    Subset nodes will be scaled such that their weight sum equals weight_frac_of_total of the sum
    over all nodes.
    """

    def __init__(
        self,
        nodes_name: str,
        nodes_attribute_name: str,
        scaling_mask_attribute_name: str,
        weight_frac_of_total: float,
        inverse: bool = False,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        self.scaling_mask_attribute_name = scaling_mask_attribute_name
        self.weight_frac_of_total = weight_frac_of_total
        super().__init__(
            nodes_name=nodes_name,
            nodes_attribute_name=nodes_attribute_name,
            inverse=inverse,
            norm=norm,
            **kwargs,
        )

    def reweight_attribute_values(self, values: torch.Tensor) -> torch.Tensor:
        scaling_mask = self.nodes[self.scaling_mask_attribute_name].squeeze()
        unmasked_sum = torch.sum(values[~scaling_mask])
        weight_per_masked_node = (
            self.weight_frac_of_total / (1 - self.weight_frac_of_total) * unmasked_sum / sum(scaling_mask)
        )
        values[scaling_mask] = weight_per_masked_node

        LOGGER.info(
            "Weight of nodes in %s rescaled such that their sum equals %.3f of the sum over all nodes",
            self.scaling_mask_attribute_name,
            self.weight_frac_of_total,
        )
        return values

    def get_scaling_values(self, **kwargs) -> torch.Tensor:
        attribute_values = super().get_scaling_values(**kwargs)
        return self.reweight_attribute_values(attribute_values)

    def on_batch_start(self, model):
        self.output_mask = model.model.model.output_mask if model.model.model.output_mask is not None else NoOutputMask()
        self.nodes = model.model.model.graph[self.nodes_name]
        if self.scaling_mask_attribute_name not in self.nodes:
            error_msg = f"scaling_mask_attribute_name {self.scaling_mask_attribute_name} not found in graph_object"
            raise KeyError(error_msg)
        scaling_values = self.get_scaling_values()
        return scaling_values