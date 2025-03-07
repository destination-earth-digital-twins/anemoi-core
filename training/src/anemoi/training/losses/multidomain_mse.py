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

import torch

from anemoi.training.losses.mse import WeightedMSELoss
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class MultiDomainMSELoss(WeightedMSELoss):
    

    def __init__(
        self,
        loss_name: str,
        node_weights: torch.Tensor | dict[torch.Tensor],
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted MSE Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """ 
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
            **kwargs,
        )
        self.loss_name = loss_name
        self.name = f"multidomain_wmse_{loss_name}"
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        graph_label: str = None, 
        squash: bool = True,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        graph_label = str(graph_label)
        return super().forward(pred, target, graph_label, squash, scalar_indices, without_scalars) if self.loss_name == graph_label else torch.Tensor([float.nan])