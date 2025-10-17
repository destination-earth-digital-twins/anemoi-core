# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# should edit copyright to both ECMWF and DestinE (I think we should...)??? 
# TODO: ASK THOMAS

import logging
from typing import Optional, Dict

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.utils import load_layer_kernels
#from anemoi.models.layers.mapper.dynamic import DynamicGraphTransformerBaseMapper
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

class DeterministicMultiDomain(AnemoiModelEncProcDec): 
    def __init__(    
        self,
        *,
        model_config: DotDict,
        data_indices: Dict[dict],
        statistics, Dict[dict],
        graph_data: Dict[HeteroData],
        truncation_data: Dict[dict], 
    ) -> None:
      """
        Initializes Determinstic Multi-Domain.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            A dictonary of data indices
        graph_data : dict[HeteroData]
            A dictonary of graph definitions
        truncation_data: Dict[dict]
            A dictonary of truncation matrices
        """
        # we dont want to inherit encprocdec cnstructor
        
        model_config = DotDict(model_config)
        self.graph_data = graph_data

        self.data_indices = data_indices
        self.statistics = statistics
        self._truncation_data = truncation_data

        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden
        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels


class EnsembleMultiDomain(DeterministicMultiDomain):
    def __init__(    
        self,
        *,
        model_config: DotDict,
        data_indices: Dict[dict],
        statistics, Dict[dict],
        graph_data: Dict[HeteroData],
        truncation_data: Dict[dict], 
    ) -> None:
      """
        Initializes Ensemble Multi-Domain.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            A dictonary of data indices
        graph_data : dict[HeteroData]
            A dictonary of graph definitions
        truncation_data: Dict[dict]
            A dictonary of truncation matrices
        """
        super().__init__(
            model_config,
            data_indices,
            statistics,
            graph_data,
            truncation_data
        ) 

    
    def forward(
        self, 
        x: torch.Tensor,
        graph_label: str,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
         """Forward pass of the model.

        Parameters
        ----------
        x : Tensor
            ensemble mode -> Input tensor, shape (bs, m, e, n, f)
            determinstic mode -> Input tensor, shape (bs, m, 1, n, f)

        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None

        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        pass 


class MultiDomain(nn.Module):
    def __init__(    
        self,
        *,
        model_config: DotDict,
        data_indices: Dict[dict],
        statistics, Dict[dict],
        graph_data: Dict[HeteroData],
        truncation_data: Dict[dict], 
    ) -> None:

    # This is the model interface
    self.model = instantiate(
        model_config.model.model_type,
        model_config,
        data_indices,
        statistics,
        graph_data,
        truncation_data, 
    )

    
    def forward(
        self, 
        x: torch.Tensor,
        graph_label: str,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
         """Forward pass of the model.

        Parameters
        ----------
        x : Tensor
            ensemble mode -> Input tensor, shape (bs, m, e, n, f)
            determinstic mode -> Input tensor, shape (bs, m, 1, n, f)

        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None

        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """

        return self.model.forward(x, graph_label, model_comm_group, grid_shard_shapes)


