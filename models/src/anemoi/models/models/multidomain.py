# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

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
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

def _embedding_indices():
    return {'q_50': 0, 'q_100': 1, 'q_150': 2, 'q_200': 3, 'q_250': 4, 'q_300': 5, 'q_400': 6, 'q_500': 7, 'q_600': 8, 'q_700': 9, 'q_850': 10, 'q_925': 11, 'q_1000': 12, 't_50': 13, 't_100': 14, 't_150': 15, 't_200': 16, 't_250': 17, 't_300': 18, 't_400': 19, 't_500': 20, 't_600': 21, 't_700': 22, 't_850': 23, 't_925': 24, 't_1000': 25, 'u_50': 26, 'u_100': 27, 'u_150': 28, 'u_200': 29, 'u_250': 30, 'u_300': 31, 'u_400': 32, 'u_500': 33, 'u_600': 34, 'u_700': 35, 'u_850': 36, 'u_925': 37, 'u_1000': 38, 'v_50': 39, 'v_100': 40, 'v_150': 41, 'v_200': 42, 'v_250': 43, 'v_300': 44, 'v_400': 45, 'v_500': 46, 'v_600': 47, 'v_700': 48, 'v_850': 49, 'v_925': 50, 'v_1000': 51, 'w_50': 52, 'w_100': 53, 'w_150': 54, 'w_200': 55, 'w_250': 56, 'w_300': 57, 'w_400': 58, 'w_500': 59, 'w_600': 60, 'w_700': 61, 'w_850': 62, 'w_925': 63, 'w_1000': 64, 'z_50': 65, 'z_100': 66, 'z_150': 67, 'z_200': 68, 'z_250': 69, 'z_300': 70, 'z_400': 71, 'z_500': 72, 'z_600': 73, 'z_700': 74, 'z_850': 75, 'z_925': 76, 'z_1000': 77, '10u': 78, '10v': 79, '2d': 80, '2t': 81, 'lsm': 82, 'msl': 83, 'sdor': 84, 'skt': 85, 'slor': 86, 'sp': 87, 'tcw': 88, 'z': 89, 'cp': 90, 'tp': 91, 'cos_latitude': 92, 'cos_longitude': 93, 'sin_latitude': 94, 'sin_longitude': 95, 'cos_julian_day': 96, 'cos_local_time': 97, 'sin_julian_day': 98, 'sin_local_time': 99, 'insolation': 100}

class AnemoiMultiDomain(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: dict[HeteroData],  # None
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        self.augment_variable = True #model_config.get("augment_variable", False)
        self._graph_data = graph_data
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden
        self.embedding_indices = _embedding_indices() #model_config.get("embedding_indices", {})
        self.union_variables = list(self.embedding_indices.keys())
        self.enable_construct_batch = True #model_config.get("enable_construct_batch", False)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.data_indices = data_indices
        self.statistics = statistics

        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        node_dim = model_config.model.node_dim  # Should inherit from config
        input_dim = self.multi_step * self.num_input_channels + node_dim

        # read config.model.layer_kernels to get the implementation for certain layers
        self.layer_kernels = load_layer_kernels(
            model_config.get("model.layer_kernels", {})
        )

        self.dynamic_embedding = instantiate(
            model_config.model.dynamic_embedding,
            data_indices=data_indices.internal_model.input,
            embedding_indices = self.embedding_indices,
            in_channels=self.num_input_channels
        )

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=, #input_dim,  # 258 -> (128*2 + 2)
            in_channels_dst=node_dim,  # 4
            hidden_dim=self.num_channels,
            layer_kernels=self.layer_kernels,
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels,
            layer_kernels=self.layer_kernels,
        )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            layer_kernels=self.layer_kernels,
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(
                    cfg,
                    name_to_index=self.data_indices.internal_model.output.name_to_index,
                    statistics=self.statistics,
                    name_to_index_stats=self.data_indices.data.input.name_to_index,
                )
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.internal_model.input)
        self.num_output_channels = len(data_indices.internal_model.output)
        self._internal_input_idx = data_indices.internal_model.input.prognostic
        self._internal_output_idx = data_indices.internal_model.output.prognostic

    def _assert_matching_indices(self, data_indices: dict) -> None:

        assert len(self._internal_output_idx) == len(
            data_indices.internal_model.output.full
        ) - len(data_indices.internal_model.output.diagnostic), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the internal output indices excluding diagnostic variables "
            f"({len(data_indices.internal_model.output.full) - len(data_indices.internal_model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        sub_graph: HeteroData,
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        return checkpoint(
            mapper,
            data,
            sub_graph=sub_graph,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            use_reentrant=use_reentrant,
        )

    def augment(self, x: Tensor) -> Tensor:
        idx = torch.randint(low=0, high=len(self.data_indices))
        x[..., idx] = 0.0
        return x 
    
    def contruct_batch(self, x: Tensor) -> Tensor:
        B,T, E, G, V = x.shape
        new_x = torch.zeros((B,T,E,G,len(self.union_variables)), x.device, dtype=x.dtype)
        for variable, index in self.embedding_indices.items():
            new_x[...,index] = x[...,self.data_indices.name_to_index[variable]]
        return new_x
    
    def missing_variables_mask(self, x: Tensor) -> Tensor:
        """Create a mask for missing variables.
        args:
            x (Tensor): Input tensor of shape (batch*grid*ensemble, vars, time)

        Returns
        -------
        Tensor
            Mask for missing variables
        """
        # Create a mask for missing variables
        presence_tensor = torch.ones(*x.shape)
        for var_idx in self.data_indices.internal_model.input.missing:
            presence_tensor[..., var_idx] = 0
        return presence_tensor

    def _assemble_input(
        self,
        x: Tensor,
        graph: HeteroData,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:
        x_data_latent = torch.cat(
            (
                einops.rearrange(
                    x,
                    "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
                ),
                torch.cat(
                    [
                        torch.sin(graph[self._graph_name_data].x),
                        torch.cos(graph[self._graph_name_data].x),
                    ],
                    dim=-1,
                ),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = torch.cat(
            [
                torch.sin(graph[self._graph_name_hidden].x),
                torch.cos(graph[self._graph_name_hidden].x),
            ],
            dim=-1,
        )

        return x_data_latent, x_hidden_latent

    def encode(
        self,
        x: tuple[Tensor, Tensor],
        graph: HeteroData,
        shard_shapes: tuple[list],
        batch_size: int,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[Tensor, Tensor]:
        x_data_latent, x_hidden_latent = x

        presence_mask = self.missing_variables_mask(x_data_latent)
        x_data_latent_embedded = self.dynamic_embedding(x_data_latent, presence_mask)

        shard_shapes_data, shard_shapes_hidden = shard_shapes
        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent_embedded, x_hidden_latent),
            sub_graph=graph[(self._graph_name_data, "to", self._graph_name_hidden)],
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        return x_data_latent, x_latent

    def decode(
        self,
        x: tuple[Tensor],
        graph: HeteroData,
        batch_size: int,
        ensemble_size: int,
        shard_shapes: list,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        shard_shapes_hidden, shard_shapes_data = shard_shapes
        x_latent_proc, x_data_latent = x

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            sub_graph=graph[(self._graph_name_hidden, "to", self._graph_name_data)],
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        # x_out = self.projection(x_out)
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x.dtype)
            .clone()
        )

    def forward(
        self,
        x: Tensor,
        graph_label: str,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        graph = self._graph_data[graph_label]
        graph = graph.to(x.device)  # this is done recurisvely under the hood

        if self.enable_construct_batch:
            x = self.contruct_batch(x)

        if self.augment_variable:
            x = self.augment(x)

        x_data_latent, x_hidden_latent = self._assemble_input(
            x, graph, model_comm_group
        )

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        x_data_latent, x_latent = self.encode(
            x,
            graph,
            shard_shapes_hidden,
            shard_shapes_data,
            batch_size=batch_size,
            midel_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            sub_graph=graph[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        x_out = self.decode(
            (x_latent_proc, x_data_latent),
            graph,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        # residual connection (just for the prognostic variables)
        # is residual connection valid when using dynamic embedding?
        x_out[..., self._internal_output_idx] += x[
            :, -1, :, :, self._internal_input_idx
        ]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)

        return x_out
