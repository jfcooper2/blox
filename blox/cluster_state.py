import sys
import time
import copy
import grpc
import json
import logging
import argparse
import pandas as pd
import numpy as np
import time
from concurrent import futures
from itertools import cycle

from typing import Tuple, List

from blox_manager import BloxManager

# from profile_parsers import pparsers


class ClusterState(object):
    """
    Keep tracks of cluster state
    """

    def __init__(self, args: argparse.ArgumentParser) -> None:
        # self.blr = blox_resource_manager
        # keeps a map of nodes
        self.server_map = dict()
        # keeps the count of node
        self.node_counter = 0
        # number of GPUs
        self.gpu_number = 0
        # types of GPU to create, cycles through as they are created
        self.gpu_types = cycle(['V100', 'P100', 'K80'])
        # gpu dataframe for easy slicing
        self.gpu_df = pd.DataFrame(
            columns=[
                "GPU_ID",
                "GPU_UUID",
                "Local_GPU_ID",
                "Node_ID",
                "IP_addr",
                "IN_USE",
                "JOB_IDS",
                "GPU_TYPE"
            ]
        )
        self.time = 0
        self.cluster_stats = dict()
        self.cluster_stats["utilization"] = []

    # def get_new_nodes(self):
    # """
    # Fetch any new nodes which have arrived at the scheduler
    # """
    # new_nodes = self.blr.rmserver.get_new_nodes()
    # return new_nodes

    def _add_new_machines(self, new_nodes: List[dict]) -> None:
        """
        Pops information of new machines and keep track of them

        new_node : list of new nodes registered with the resource manager
        """
        while True:
            try:
                node_info = new_nodes.pop(0)
                self.server_map[self.node_counter] = node_info
                numGPUs_on_node = node_info["numGPUs"]
                gpu_uuid_list = node_info["gpuUUIDs"].split("\n")
                assert (
                    len(gpu_uuid_list) == numGPUs_on_node
                ), f"GPU UUIDs {len(gpu_uuid_list)}  GPUs on node {numGPUs_on_node}"
                if numGPUs_on_node > 0:
                    gpuID_list = list()
                    for local_gpu_id, gpu_uuid in zip(
                        range(numGPUs_on_node), gpu_uuid_list
                    ):
                        gpu_type = next(self.gpu_types)
                        gpuID_list.append(
                            {
                                "GPU_ID": self.gpu_number,
                                "Node_ID": self.node_counter,
                                "GPU_UUID": gpu_uuid,
                                "Local_GPU_ID": local_gpu_id,
                                "IP_addr": node_info["ipaddr"],
                                "IN_USE": False,
                                "JOB_IDS": None,
                                "GPU_TYPE": gpu_type
                            }
                        )
                        self.gpu_number += 1
                    self.gpu_df = self.gpu_df.append(gpuID_list)
                    self.node_counter += 1
            except IndexError:
                break

    def update(self, new_nodes):
        """
        Updates cluster state by fetching new nodes
        Args:
            None
        Returns:
            new_nodes : List of new nodes
        """
        # getting new updates
        # new_nodes = self.blr.rmserver.get_new_nodes()
        self.cluster_stats["utilization"].append((self.time, np.sum(self.gpu_df["IN_USE"])))

        if len(new_nodes) > 0:
            self._add_new_machines(new_nodes)
        return new_nodes
