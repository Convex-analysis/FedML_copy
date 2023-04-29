import logging
import numpy as np
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
        if total_canadiate_client == client_num_per_round:
            client_indexes = [client_index for client_index in range(total_canadiate_client)]
        else:
            num_clients = min(client_num_per_round, total_canadiate_client)
            np.random.seed(global_round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(total_canadiate_client), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        
        group_to_client_indexes = {}
        for client_idx in client_indexes:
            group_idx = self.group_indexes[client_idx]
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        logging.info(
            "client_indexes of each group = {}".format(group_to_client_indexes)
        )
        return client_indexes
    

            

"""

"""
