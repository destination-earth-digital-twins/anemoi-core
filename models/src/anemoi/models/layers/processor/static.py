# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Optional

from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.khop_edges import sort_edges_1hop_sharding
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.chunk import GNNProcessorChunk
from anemoi.models.layers.chunk import GraphTransformerProcessorChunk
from anemoi.models.layers.chunk import TransformerProcessorChunk
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.mapper.base import GraphEdgeMixin
from anemoi.models.layers.processor.base import BaseProcessor
from anemoi.utils.config import DotDict


class TransformerProcessor(BaseProcessor):
    """Transformer Processor."""

    def __init__(
        self,
        num_layers: int,
        layer_kernels: DotDict,
        *args,
        window_size: Optional[int] = None,
        num_channels: int = 128,
        num_chunks: int = 2,
        activation: str = "GELU",
        cpu_offload: bool = False,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        dropout_p: float = 0.1,
        attention_implementation: str = "flash_attention",
        softcap: float = 0.0,
        use_alibi_slopes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize TransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of num_layers
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels['Linear'] = "torch.nn.Linear"
            Defined in config/models/<model>.yaml
        window_size: int,
            1/2 size of shifted window for attention computation
        num_channels : int
            number of channels
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        dropout_p: float, optional
            Dropout probability used for multi-head self attention, default 0.0
        attention_implementation: str, optional
            A predefined string which selects which underlying attention
            implementation, by default "flash_attention"
        softcap : float, optional
            Anything > 0 activates softcapping flash attention, by default None
        use_alibi_slopes : bool, optional
            Use aLiBI option, only used for flash attention, by default None
        """
        super().__init__(
            num_layers=num_layers,
            num_channels=num_channels,
            window_size=window_size,
            num_chunks=num_chunks,
            activation=activation,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
        )

        self.build_layers(
            TransformerProcessorChunk,
            num_channels=num_channels,
            num_layers=self.chunk_size,
            layer_kernels=layer_kernels,
            mlp_hidden_ratio=mlp_hidden_ratio,
            num_heads=num_heads,
            window_size=window_size,
            activation=activation,
            dropout_p=dropout_p,
            attention_implementation=attention_implementation,
            softcap=softcap,
            use_alibi_slopes=use_alibi_slopes,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], ...],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        (x,) = self.run_layers((x,), shape_nodes, batch_size, model_comm_group)

        return x


class GNNProcessor(GraphEdgeMixin, BaseProcessor):
    """GNN Processor."""

    def __init__(
        self,
        num_layers: int,
        layer_kernels: DotDict,
        *args,
        trainable_size: int = 8,
        num_channels: int = 128,
        num_chunks: int = 2,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        cpu_offload: bool = False,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
        **kwargs,
    ) -> None:
        """Initialize GNNProcessor.

        Parameters
        ----------
        num_channels : int
            Number of Channels
        num_layers : int
            Number of layers
        num_chunks : int, optional
            Number of num_chunks, by default 2
        mlp_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            activation=activation,
            cpu_offload=cpu_offload,
            mlp_extra_layers=mlp_extra_layers,
        )

        self._register_edges(
            sub_graph,
            sub_graph_edge_attributes,
            src_grid_size,
            dst_grid_size,
            trainable_size,
        )

        self.trainable = TrainableTensor(
            trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0]
        )

        kwargs = {
            "mlp_extra_layers": mlp_extra_layers,
            "activation": activation,
            "edge_dim": None,
        }

        self.build_layers(
            GNNProcessorChunk, num_channels, self.chunk_size, layer_kernels, **kwargs
        )

        kwargs["edge_dim"] = self.edge_dim  # Edge dim for first layer
        self.proc[0] = GNNProcessorChunk(
            num_channels, self.chunk_size, layer_kernels, **kwargs
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)
        target_nodes = sum(x[0] for x in shape_nodes)
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = (
            sort_edges_1hop_sharding(
                target_nodes,
                edge_attr,
                edge_index,
                model_comm_group,
            )
        )
        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            (x, edge_attr), edge_index, (shape_nodes, shape_nodes), model_comm_group
        )

        return x


class GraphTransformerProcessor(GraphEdgeMixin, BaseProcessor):
    """Processor."""

    def __init__(
        self,
        num_layers: int,
        layer_kernels: DotDict,
        trainable_size: int = 8,
        num_channels: int = 128,
        num_chunks: int = 2,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        cpu_offload: bool = False,
        sub_graph: Optional[HeteroData] = None,
        sub_graph_edge_attributes: Optional[list[str]] = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
        **kwargs,
    ) -> None:
        """Initialize GraphTransformerProcessor.

        Parameters
        ----------
        num_layers : int
            Number of layers
        num_channels : int
            Number of channels
        num_chunks : int, optional
            Number of num_chunks, by default 2
        heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        """
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            num_chunks=num_chunks,
            activation=activation,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
        )

        self._register_edges(
            sub_graph,
            sub_graph_edge_attributes,
            src_grid_size,
            dst_grid_size,
            trainable_size,
        )

        self.trainable = TrainableTensor(
            trainable_size=trainable_size, tensor_size=self.edge_attr.shape[0]
        )

        self.build_layers(
            GraphTransformerProcessorChunk,
            num_channels=num_channels,
            num_layers=self.chunk_size,
            layer_kernels=layer_kernels,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            activation=activation,
            edge_dim=self.edge_dim,
        )

        self.offload_layers(cpu_offload)

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        shape_nodes = change_channels_in_shape(shard_shapes, self.num_channels)
        edge_attr = self.trainable(self.edge_attr, batch_size)

        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x, edge_attr = self.run_layers(
            (x, edge_attr),
            edge_index,
            (shape_nodes, shape_nodes, shapes_edge_attr),
            batch_size,
            model_comm_group,
        )

        return x
