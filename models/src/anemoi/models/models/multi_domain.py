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
from anemoi.models.models import AnemoiModelEncProcDec, AnemoiEnsDatasetsDataModule
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

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )

    def _calculate_input_dim(self):
        # overwrite base method
        # TODO: investigate in depth
        return self.multi_step * self.num_input_channels + self.node_dim

    def _calculate_input_dim_latent(self):
        # overwrite base method
        # TODO: investigate in depth
        return self.node_dim

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.input_dim,
            in_channels_dst=self.input_dim_latent,
            hidden_dim=self.num_channels,
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.num_channels,
        )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.num_channels,
            in_channels_dst=self.input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
        )

    def _assemble_input(
        self, 
        x: torch.Tensor, 
        graph: HeteroData, 
        batch_size: int, 
        grid_shard_shapes: Optional[tuple]=None, 
        model_comm_group: Optional[ProcessGroup]=None
        ) -> torch.Tensor:
        x_skip = x[:, -1, ...]
        x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
        # TODO: find a solution for truncation matrix!
        x_skip = self.truncation(x_skip, grid_shard_shapes, model_comm_group)
        x_skip = einops.rearrange(x_skip, "(batch ensemble) grid vars -> batch ensemble grid vars", batch=batch_size)

        node_attributes_data = torch.cat(
                [
                    torch.sin(graph[self._graph_name_data].x), 
                    torch.cos(graph[self._graph_name_data].x)
                ], 
                dim = -1
            )
        
        # TODO: investigate if this correct and gives correct shapes...
        node_attributes_data = einops.repeat(
            node_attributes_data, 
            "grid -> grid batch_size", 
            batch_size=batch_size
        )

        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_hidden_latent(
        self,
        x: torch.Tensor, 
        graph: HeteroData, 
        batch_size: int, 
        model_comm_group: Optional[ProcessGroup] = None
        ) -> tuple[torch.Tensor, list]:

        x_hidden_latent = torch.cat(
            [
                torch.sin(graph[self._graph_name_hidden].x), 
                torch.cos(graph[self._graph_name_hidden].x)
            ], 
            dim = -1
        )

        x_hidden_latent = einops.repeat(
            x_hidden_latent, 
            "grid -> grid batch_size", 
            batch_size=batch_size
        )

        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group=model_comm_group)

        return x_hidden_latent, shard_shapes_hidden

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        x: torch.Tensor, 
        graph_label: str, 
        *,
        fcaststep: Optional[int] = None, # <--- deterministic does not use it, just place holder
        model_comm_group: Optional[ProcessGroup]= None,
        grid_shard_shapes: Optional[tuple] = None,
        **kwargs,
        ) -> torch.Tensor:

        batch_size = batch.shape[0]
        ensemble_size = batch.shape[2]

        graph = self.graph_data[graph_label]
        graph.to(x.device)

        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        x_hidden_latent, sharp_shapes_hiddem = self._assemble_hidden_latent(
            x, graph, batch_size, model_comm_group,
        )

        # Encoder
        x_data_latent, x_latent = self.encoder(
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            sub_graph=graph[(self._graph_name_data, "to", self._graph_name_hidden)],
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        # Processor
        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            sub_graph=graph[(self._graph_name_hidden, "to", self._graph_name_hidden)]
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # Skip
        x_latent_proc = x_latent_proc + x_latent

        # Decoder
        x_out = self.decoder(
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            sub_graph=graph[(self._graph_name_hidden, "to", self._graph_name_data)]
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out



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
    def _calculate_input_dim(self):
        # TODO: ensure this is correct!!!!
        base_input_dim = super()._calculate_input_dim()
        return base_input_dim + self.num_input_channels_prognostic + 1

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""
        super()._build_networks(model_config)
        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False, 
            num_channels=self.num_channels,
        )
    
    def _assemble_input(
        self, 
        x: torch.Tensor, 
        graph: HeteroData,
        fcstep: int, 
        bse: int, 
        grid_shard_shapes: Optional[tuple]=None, 
        model_comm_group: Optional[ProcessGroup]=None
        )-> torch.Tensor:
        x_skip = x[:, -1, :, :, self._internal_input_idx]
        x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
        # TODO: find a solution for truncation matrix!
        x_skip = self.truncation(x_skip, grid_shard_shapes, model_comm_group)

        node_attributes_data = torch.cat(
                [
                    torch.sin(graph[self._graph_name_data].x), 
                    torch.cos(graph[self._graph_name_data].x)
                ], 
                dim = -1
            )

        # TODO: investigate if this correct and gives correct shapes...
        node_attributes_data = einops.repeat(
            node_attributes_data, 
            "grid -> grid bse", 
            bse=bse
        )

        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        x_data_latent = torch.cat(
            (x_data_latent, torch.ones(x_data_latent.shape[:-1], device=x_data_latent.device).unsqueeze(-1) * fcstep),
            dim=-1,
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, bse, dtype):
        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=batch_size).to(dtype=dtype).clone()

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += einops.rearrange(
            x_skip,
            "(batch ensemble) grid var -> batch ensemble grid var",
            batch=batch_size,
        ).to(dtype=dtype)

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        self, 
        x: torch.Tensor,
        graph_label: str,
        *,
        fcststep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward operator.

        Args:
            x: torch.Tensor
                Input tensor, shape (bs, m, e, n, f)
            fcstep: int
                Forecast step
            model_comm_group: Optional[ProcessGroup], optional
                Model communication group
            grid_shard_shapes : list, optional
                Shard shapes of the grid, by default None
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor
        """
        batch_size, ensemble_size = x.shape[0], x.shape[2]
        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        graph = self._graph_data[graph_label]
        graph.to(x.device)

        fcstep = min(1, fcstep)

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, graph, fcstep, bse, grid_shard_shapes, model_comm_group
        )

        x_hidden_latent, shard_shapes_hidden = super()._assemble_hidden_latent(
            x, graph, bse, model_comm_group
        )

        x_data_latent, x_latent = self.encoder(
            (x_data_latent, x_hidden_latent),
            batch_size=bse,
            sub_graph=graph[(self._graph_name_data, "to", self._graph_name_hidden)],
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            noise_ref=x_hidden_latent,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=bse,
            sub_graph=graph[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )

        x_latent_proc = x_latent_proc + x_latent

        x_out = self.decoder(
            (x_latent_proc, x_data_latent),
            batch_size=bse,
            sub_graph=graph[(self._graph_name_hidden, "to", self._graph_name_data)],
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, bse, x.dtype)

        return x_out


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
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
        truncation_data=truncation_data, 
    )

    
    def forward(
        self, 
        x: torch.Tensor,
        graph_label: str,
        *,
        fcstep: Optional[int] = None, 
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
         """
         Forward pass of the model.

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

        return self.model.forward(
            x, 
            graph_label, 
            fcststep,
            model_comm_group, 
            grid_shard_shapes
            )


