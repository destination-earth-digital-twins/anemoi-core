# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj
from torch_geometric.typing import PairTensor
from torch_geometric.utils import bipartite_subgraph


from anemoi.models.distributed.graph import shard_tensor, sync_tensor, gather_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape, get_shape_shards
from anemoi.models.distributed.khop_edges import drop_unconnected_src_nodes,sort_edges_1hop_sharding
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.mapper.base import BackwardMapperPostProcessMixin
from anemoi.models.layers.mapper.base import BaseMapper
from anemoi.models.layers.mapper.base import ForwardMapperPreProcessMixin
from anemoi.models.layers.mlp import MLP
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class DynamicGraphTransformerBaseMapper(BaseMapper):
    """Dynamic Graph Transformer Base Mapper from hidden -> data or data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        out_channels_dst: Optional[int] = None,
        sub_graph_edge_attributes: Optional[list] = [],
        sub_graph_edge_index_name: str = "edge_index",
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "GELU",
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        edge_dim: int = 0,
        layer_kernels: DotDict = None,
        shard_strategy: str = "edges"
    ) -> None:
        """Initialize DynamicGraphTransformerBaseMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        sub_graph_edge_attributes: list[str]
            Names of edge attributes to consider
        sub_graph_edge_index_name: str
            Name of the edge index attribute in the graph. Defaults to "edge_index".
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        edge_dim : int, optional
            The dimension of the edge attributes
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
        )
        #print(num_chunks, "num chunks in dynamic mapper")
        #print(cpu_offload, "cpu offload in dynamic mapper")
        self.num_chunks = num_chunks
        self.edge_attribute_names = sub_graph_edge_attributes
        self.edge_index_name = sub_graph_edge_index_name

        self.proc = GraphTransformerMapperBlock(
            hidden_dim,
            mlp_hidden_ratio * hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            activation=activation,
            num_chunks=num_chunks,
            layer_kernels=layer_kernels,
        )

        self.offload_layers(cpu_offload)
        # idst 4, h 128, 
        """self.emb_nodes_dst = (
            nn.Linear(self.in_channels_dst, self.hidden_dim)
            if self.in_channels_dst != self.hidden_dim
            else nn.Identity()
        )"""
        Linear = layer_kernels["Linear"]
        self.emb_nodes_dst = Linear(self.in_channels_dst, self.hidden_dim)
        
        self.shard_strategy = shard_strategy
        print(self.shard_strategy)
        assert shard_strategy in ["heads", "edges"], (
            f"Invalid shard strategy '{shard_strategy}' for {self.__class__.__name__}. "
            f"Supported strategies are 'heads' and 'edges'."
        )

    def prepare_edges(
        self,
        size: tuple[int, int],
        batch_size: int,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        #edge_inc: int,
        model_comm_group: Optional[ProcessGroup] = None,
    )->tuple[torch.Tensor, Adj]:
        #edge_attr = self.egde_attr
        #edge_index = torch.cat(
        #    [edge_index + i * edge_inc for i in range(batch_size)],
        #    dim=1,
        #)
        #print(edge_attr.shape, "before 1hop")
        #print(edge_index.shape, "before 1hop")
        edge_attr, edge_index, shapes_edge_attr, shapes_edge_idx = sort_edges_1hop_sharding(
            size, edge_attr, edge_index, model_comm_group, relabel_dst_nodes=True
        )
        #print(edge_attr.shape, "after 1hop")
        #print(edge_index.shape, "after 1hop")
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)
        edge_index = shard_tensor(edge_index, 1, shapes_edge_idx, model_comm_group)
        #print(edge_attr.shape, "after sharding edge_attr")
        #print(edge_index.shape, "after sharding edge_index")
        return edge_attr, edge_index

    def pre_process_edge_sharding_wrapper(
        self,
        x: PairTensor,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        #edge_inc: int,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
    ):
        x_src, x_dst = x
        shapes_src, shapes_dst = shard_shapes

        shapes_x_src = change_channels_in_shape(shapes_src, x_src.shape[-1])
        # gather/scatter if x_src is sharded, always reduce gradients in bwds
        x_src = sync_tensor(x_src, 0, shapes_x_src, model_comm_group, gather_in_fwd=x_src_is_sharded)
        
        # full size of the graph
        size_full_graph = (sum(shape[0] for shape in shard_shapes[0]), sum(shape[0] for shape in shard_shapes[1]))
        edge_attr, edge_index = self.prepare_edges(
            size=size_full_graph, 
            batch_size=batch_size,
            edge_attr=edge_attr,
            edge_index=edge_index,
            #edge_inc=edge_inc, 
            model_comm_group=model_comm_group
        )

         # At this point, x_src is synced i.e. full, x_dst is sharded, edges are sharded (incoming edges to x_dst)
        size_src_full_dst_shard = (x_src.shape[0], x_dst.shape[0])
        x_src, edge_index = drop_unconnected_src_nodes(x_src, edge_index, size_src_full_dst_shard)

        if not x_dst_is_sharded:
            x_dst= shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
        
        return x_src, x_dst, edge_attr, edge_index, shapes_src, shapes_dst

    def run_processor_chunk_edge_sharding(
        self,
        x: tuple[torch.Tensor,torch.Tensor],
        dst_chunk: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        shapes: tuple[tuple[int], tuple[int]],
        batch_size: int,
        size: tuple[int],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> torch.Tensor:

        x_src, x_dst = x
        # get subgraph of x_dst_chunk and incoming edges, drop unconnected src nodes
        nodes_src_full = torch.arange(size[0], device=edge_index.device)
        edge_index, edge_attr = bipartite_subgraph(
            (nodes_src_full, dst_chunk),
            edge_index,
            edge_attr,
            size=size,
            relabel_nodes=True,
        )
        # drop unconnected src nodes and relabel edges
        x_src_chunk, edge_index_chunk = drop_unconnected_src_nodes(x_src, edge_index, size)
        x_dst_chunk = x_dst[dst_chunk]
        chunk_size = (x_src_chunk.shape[0], x_dst_chunk.shape[0])

        # pre-process chunk, embedding x_src and x_dst if not already done
        x_src_chunk, x_dst_chunk, _, _ = self.pre_process(
            (x_src_chunk, x_dst_chunk),shapes, model_comm_group, x_src_is_sharded=True, x_dst_is_sharded=True)
        
        
        (_, x_dst_out), _ = self.proc(
            x=(x_src_chunk, x_dst_chunk),
            edge_attr=edge_attr,
            edge_index=edge_index_chunk,
            shapes=shapes,
            batch_size=batch_size,
            size=chunk_size,
            model_comm_group=model_comm_group,
        )
        return self.post_process(x_dst_out, shapes[1], model_comm_group, keep_x_dst_sharded=True)
            
    
    def edge_sharding(
        self, 
        x: PairTensor,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        #edge_inc: int,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
    )-> PairTensor:
        x_src, x_dst, edge_attr, edge_index, shapes_src, shapes_dst = checkpoint(
            self.pre_process_edge_sharding_wrapper,
            x=x,
            edge_attr=edge_attr,
            edge_index=edge_index,
            #edge_inc=edge_inc,
            shard_shapes=shard_shapes,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
            x_src_is_sharded=x_src_is_sharded,
            x_dst_is_sharded=x_dst_is_sharded,
            use_reentrant=False,
        )

        size = (x_src.shape[0], x_dst.shape[0]) # Node sizes of local graph sharded
        #num_chunks = max(self.num_chunks, 1
        
        dst_chunks = torch.arange(size[1], device=x_dst.device).tensor_split(self.num_chunks)
        out_channels = self.out_channels_dst if self.out_channels_dst is not None else self.hidden_dim
        out_type = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x_dst.dtype
        out_dst = torch.empty((*x_dst.shape[:-1], out_channels), dtype=out_type, device=x_dst.device)

        # run processor on each chunk
        for dst_chunk in dst_chunks:
            out_dst[dst_chunk] = checkpoint(
                self.run_processor_chunk_edge_sharding,
                (x_src, x_dst),
                dst_chunk,
                edge_attr,
                edge_index,
                (shapes_src, shapes_dst),
                batch_size,
                size,
                model_comm_group,
                use_reentrant=False,
            ).to(dtype=out_type)
        
        # gather after processing chunks
        if not keep_x_dst_sharded:
            out_dst = gather_tensor(
                out_dst, 0, change_channels_in_shape(shapes_dst, out_channels), model_comm_group,
            )
        
        return out_dst

    def heads_sharding(
        self,
        x: PairTensor,
        edge_attr: torch.Tensor,
        edge_index: Adj,
        #edge_inc: int,
        batch_size: int,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
    ) -> PairTensor:
        size = (sum(x[0] for x in shard_shapes[0]), sum(x[0] for x in shard_shapes[1]))
        #edge_index = torch.cat(
        #    [edge_index + i * edge_inc for i in range(batch_size)],
        #    dim=1,
        #)
        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(
            x, shard_shapes, model_comm_group, x_src_is_sharded, x_dst_is_sharded
        )

        (x_src, x_dst), edge_attr = self.proc(
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            edge_index=edge_index,
            shapes=(shapes_src, shapes_dst, shapes_edge_attr),
            batch_size=batch_size,
            size=size,
            model_comm_group=model_comm_group,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group, keep_x_dst_sharded=keep_x_dst_sharded)

        return x_dst
    
    def forward(
        self, 
        x: PairTensor,
        batch_size: int,
        sub_graph: HeteroData,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
    ) -> PairTensor:

        edge_index = sub_graph[self.edge_index_name].to(torch.int64)
        edge_attr = torch.cat(
            [sub_graph[attr] for attr in self.edge_attribute_names], axis=1
        )

        if self.shard_strategy == "edges":
            return self.edge_sharding(
                x=x, 
                batch_size=batch_size, 
                shard_shapes=shard_shapes, 
                edge_attr=edge_attr,
                edge_index=edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=x_src_is_sharded, 
                x_dst_is_sharded=x_dst_is_sharded, 
                keep_x_dst_sharded=keep_x_dst_sharded
            )
        else:
            return self.heads_sharding(
                x=x, 
                edge_attr=edge_attr, 
                edge_index=edge_index, 
                batch_size=batch_size, 
                shard_shapes=shard_shapes, 
                model_comm_group=model_comm_group,
                x_src_is_shared=x_src_is_sharded, 
                x_dst_is_sharded=x_dst_is_sharded, 
                keep_x_dst_sharded=keep_x_dst_sharded
            )
    def _forward(
        self,
        x: PairTensor,
        batch_size: int,
        sub_graph: HeteroData,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> PairTensor:
        #print("inside dynamic basemapper, subgraph",sub_graph.device )
        size = (sum(x[0] for x in shard_shapes[0]), sum(x[0] for x in shard_shapes[1]))
        edge_index = sub_graph[self.edge_index_name].to(torch.int64)
        edge_attr = torch.cat(
            [sub_graph[attr] for attr in self.edge_attribute_names], axis=1
        )

        shapes_edge_attr = get_shape_shards(edge_attr, 0, model_comm_group)
        edge_attr = shard_tensor(edge_attr, 0, shapes_edge_attr, model_comm_group)

        x_src, x_dst, shapes_src, shapes_dst = self.pre_process(
            x, shard_shapes, model_comm_group
        )

        (x_src, x_dst), edge_attr = self.proc(
            (x_src, x_dst),
            edge_attr,
            edge_index,
            (shapes_src, shapes_dst, shapes_edge_attr),
            batch_size,
            model_comm_group,
            size=size,
        )

        x_dst = self.post_process(x_dst, shapes_dst, model_comm_group)

        return x_dst


class DynamicGraphTransformerForwardMapper(
    ForwardMapperPreProcessMixin, DynamicGraphTransformerBaseMapper
):
    """Dynamic Graph Transformer Mapper from data -> hidden."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        out_channels_dst: Optional[int] = None,
        sub_graph_edge_attributes: Optional[list] = [],
        sub_graph_edge_index_name: str = "edge_index",
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "GELU",
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        edge_dim: int = 0,
        layer_kernels: DotDict = None,
        **kwargs,
    ) -> None:
        """Initialize DynamicGraphTransformerForwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        sub_graph_edge_attributes: list[str]
            Names of edge attributes to consider
        sub_graph_edge_index_name: str
            Name of the edge index attribute in the graph. Defaults to "edge_index".
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        edge_dim: int, optional
            Dimension of the edge attributes
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst=out_channels_dst,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            sub_graph_edge_index_name=sub_graph_edge_index_name,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
            layer_kernels=layer_kernels,
        )

        #self.emb_nodes_src = nn.Identity()
        self.emb_nodes_src = layer_kernels["Linear"](self.in_channels_src, self.hidden_dim)

    def forward(
        self,
        x: PairTensor,
        batch_size: int,
        sub_graph: HeteroData,
        shard_shapes: tuple[tuple[int], tuple[int]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
    ) -> PairTensor:
        x_dst = super().forward(
            x, batch_size, sub_graph, shard_shapes, model_comm_group,
            x_src_is_sharded=x_src_is_sharded,x_dst_is_sharded=x_dst_is_sharded, keep_x_dst_sharded=keep_x_dst_sharded,
        )
        #print("inside dynamic forward mapper")
        return x[0], x_dst


class DynamicGraphTransformerBackwardMapper(
    BackwardMapperPostProcessMixin, DynamicGraphTransformerBaseMapper
):
    """Dynamic Graph Transformer Mapper from hidden -> data."""

    def __init__(
        self,
        in_channels_src: int = 0,
        in_channels_dst: int = 0,
        hidden_dim: int = 128,
        out_channels_dst: Optional[int] = None,
        sub_graph_edge_attributes: Optional[list] = [],
        sub_graph_edge_index_name: str = "edge_index",
        num_chunks: int = 1,
        cpu_offload: bool = False,
        activation: str = "GELU",
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        edge_dim: int = 0,
        layer_kernels: DotDict = None,
        **kwargs,
    ) -> None:
        """Initialize DynamicGraphTransformerBackwardMapper.

        Parameters
        ----------
        in_channels_src : int
            Input channels of the source node
        in_channels_dst : int
            Input channels of the destination node
        hidden_dim : int
            Hidden dimension
        num_heads: int
            Number of heads to use, default 16
        mlp_hidden_ratio: int
            ratio of mlp hidden dimension to embedding dimension, default 4
        sub_graph_edge_attributes: list[str]
            Names of edge attributes to consider
        sub_graph_edge_index_name: str
            Name of the edge index attribute in the graph. Defaults to "edge_index".
        activation : str, optional
            Activation function, by default "GELU"
        cpu_offload : bool, optional
            Whether to offload processing to CPU, by default False
        out_channels_dst : Optional[int], optional
            Output channels of the destination node, by default None
        edge_dim: int, optional
            Dimension of the edge attributes
        """
        super().__init__(
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst=out_channels_dst,
            sub_graph_edge_attributes=sub_graph_edge_attributes,
            sub_graph_edge_index_name=sub_graph_edge_index_name,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            activation=activation,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            edge_dim=edge_dim,
            layer_kernels=layer_kernels,
        )

        self.node_data_extractor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_channels_dst),
        )

    def pre_process(self, x, shard_shapes, model_comm_group=None, x_src_is_sharded: bool = False, x_dst_is_sharded: bool = False):
        x_src, x_dst, shapes_src, shapes_dst = super().pre_process(
            x, shard_shapes, model_comm_group
        )
        shapes_src = change_channels_in_shape(shapes_src, self.hidden_dim)
        if not x_dst_is_sharded:
            x_dst = shard_tensor(x_dst, 0, shapes_dst, model_comm_group)
        x_dst = self.emb_nodes_dst(x_dst)
        shapes_dst = change_channels_in_shape(shapes_dst, self.hidden_dim)
        return x_src, x_dst, shapes_src, shapes_dst
