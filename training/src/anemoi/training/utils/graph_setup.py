import logging
from pathlib import Path
from typing import Optional, Dict

import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.nodes import ZarrDatasetNodes
from anemoi.graphs.utils import get_distributed_device
from anemoi.training.schemas.base_schema import convert_to_omegaconf

LOGGER = logging.getLogger(__name__)


def dist_info(len_labels: int):
    pass


def single_graph_setup(config: DictConfig) -> HeteroData:
    if config.hardware.files.graph is not None:
        graph_filename = Path(
            config.hardware.paths.graph,
            config.hardware.files.graph,
        )

        if graph_filename.exists() and not config.graph.overwrite:
            from anemoi.graphs.utils import get_distributed_device

            LOGGER.info("Loading graph data from %s", graph_filename)
            return torch.load(
                graph_filename,
                map_location=get_distributed_device(),
                weights_only=False,
            )

    else:
        graph_filename = None

    graph_config = convert_to_omegaconf(config).graph
    return GraphCreator(config=graph_config).create(
        save_path=graph_filename,
        overwrite=config.graph.overwrite,
    )


def multi_graph_setup(config: DictConfig) -> Dict[HeteroData]:
    """
    revisit. Try to explore this:

    consider a set of ranks ranks = {0,1,2...,N-1}
    a subset of ranks belonging to group G = {0,1,2,...,M-1}
    and having len(labels) <= M - 1, each group should create a graph
    in parallel and then do a broad cast so that all local ranks {0,1,2,...,N-1}
    has access to the graphs.
    """
    _graph_data = {}
    for label, data in config.dataloader.training.items():
        graph_filename = Path(
            config.hardware.paths.graph,
            label + ".pt",
        )

        if graph_filename.exist() and not config.graph.overwrite:
            LOGGER.info(f"Loading graph from {graph_filename}")
            graph = torch.load(
                graph_filename,
                map_location=get_distributed_device(),
                weights_only=False,
            )
        else:
            graph_config = convert_to_omegaconf(config).graph
            graph = ZarrDatasetNodes(data, name=config.graph.data).update_graph(
                HeteroData(), attrs_config=graph_config.attributes.nodes
            )  # empty graph

            gc = GraphCreator(config=graph_config)
            graph = gc.update_graph(graph)
            graph = gc.clean(graph)

            gc.save(graph, graph_filename, overwrite=True)

        # insert graph_label into graph obj
        graph.label = label
        _graph_data[label] = graph
    return _graph_data


def graph_setup(config: DictConfig, dynamic_mode: Optional[bool] = False) -> HeteroData:
    if dynamic_mode:
        LOGGER.info(f"dynamic_mode: {dynamic_mode}, proceeding with multiple graphs")
        return multi_graph_setup(config)

    LOGGER.info(f"dynamic_mode: {dynamic_mode}, proceeding with single graph")
    return single_graph_setup(config)
