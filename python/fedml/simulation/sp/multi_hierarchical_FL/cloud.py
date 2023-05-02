import logging
import numpy as np
from .group import Group
from ..hierarchical_fl.trainer import HierarchicalTrainer


class Cloud(HierarchicalTrainer):
    def __init__(self, args, device, model, model_trainer,client_list):
        self.args = args
        self.device = device
        self.model = model
        self.model_trainer = model_trainer
        self.group_list = None
        self.group_indexes = None
        self.client_list = client_list
    def set_group_list(self, group_list=list):
        self.group_list = group_list
    def set_group_indexes(self, group_object_list=list):
        client_to_group_indexes = {}
        for group in group_object_list:
            for client in group.client_dict.keys():
                client_to_group_indexes[client] = group.idx
        self.group_indexes = client_to_group_indexes

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
                self.group_list,
                self.args.client_num_per_round,
            )
            #train each group
            w_groups_dict = {}
            for group in self.group_list:
                sampled_client_indexes = [client for client in group_to_client_indexes if client in group.client_dict.keys()]
                w_group_list = group.train(
                    global_round_idx, w_global, sampled_client_indexes
                )
                for global_epoch, w in w_group_list:
                    if not global_epoch in w_groups_dict:
                        w_groups_dict[global_epoch] = []
                    w_groups_dict[global_epoch].append(
                        (group.get_sample_number(sampled_client_indexes), w)
                    )
            # aggregate group weights into the global weight
            for global_epoch in sorted(w_groups_dict.keys()):
                w_groups = w_groups_dict[global_epoch]
                w_global = self._aggregate(w_groups)

                # evaluate performance
                if (
                        global_epoch % self.args.frequency_of_the_test == 0
                        or global_epoch
                        == self.args.comm_round
                        * self.args.group_comm_round
                        * self.args.epochs
                        - 1
                ):
                    self.model.load_state_dict(w_global)
                    self._local_test_on_all_clients(global_epoch)


    def _client_sampling(self, global_round_idx, group_list, client_num_per_round):
        #store all groups' client and ready to sample
        total_canadiate_client = []
        for group in group_list:
            total_canadiate_client.extend(group.client_dict.keys())
        if total_canadiate_client == client_num_per_round:
            client_indexes = [client_index for client_index in range(total_canadiate_client)]
        else:
            num_clients = min(client_num_per_round, len(total_canadiate_client))
            np.random.seed(global_round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(total_canadiate_client, num_clients, replace=False)
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
