import logging

from .group import Group
from ..hierarchical_fl.trainer import HierarchicalTrainer


class Cloud(HierarchicalTrainer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.group_list = None
    def set_group_list(self, group_list=list):
        self.group_list = group_list

    def train(self):
        w_global = self.model.state_dict()
        for global_round_idx in range(self.args.comm_round):
            logging.info(
                "################Global Communication Round : {}".format(
                    global_round_idx
                )
            )
            group_to_client_indexes = self._client_sampling(
                global_round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

    def _client_sampling(self, global_round_idx, group_list, client_num_per_round):
        #store all groups' client and ready to sample
        total_canadiate_client = []
        for group in group_list:
            total_canadiate_client.extend(group.client_dict.keys())

            

"""

"""
