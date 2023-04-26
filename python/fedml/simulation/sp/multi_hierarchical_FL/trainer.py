import logging

import numpy as np

from .client import HFLClient
from .cloud import Cloud
from .group import Group
from ..fedavg.fedavg_api import FedAvgAPI


class MultiHierFLTrainer(FedAvgAPI):
    def __init__(self, args, device, comm=None, model_trainer=None):
        super().__init__(args, device, comm, model_trainer)
        self.args = args
        self.device = device
        self.comm = comm
        self.model_trainer = model_trainer
        self.model = model_trainer.model
        self.group_dict = None
        self.group_indexes = None
        self.client_list = None
        self._setup_clients_for_MHFL(
            self.train_data_local_num_dict,
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.model_trainer,
        )
        self.federation_list = self._setup_federations(self.args.federation_num)

    #setting up federations in our simulation
    def _setup_federations(self, federation_num):
        list_of_federation = []
        for i in range(federation_num):
            list_of_federation.append(Cloud(self.args, self.device, self.model, self.model_trainer))
        return list_of_federation

    #resign all client to a corresponding group
    def _setup_clients_for_MHFL(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == "random":
            self.group_indexes = np.random.randint(
                0, self.args.group_num, self.args.client_num_in_total
            )
            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)
        else:
            raise Exception(self.args.group_method)

        self.group_dict = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            self.group_dict[group_idx] = Group(
                group_idx,
                client_indexes,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                self.args,
                self.device,
                self.model,
                self.model_trainer
            )

        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        client_idx = -1
        self.client_list = [
            HFLClient(
                client_idx,
                train_data_local_dict[0],
                test_data_local_dict[0],
                train_data_local_num_dict[0],
                self.args,
                self.device,
                self.model,
                self.model_trainer
            )
        ]
        logging.info("############setup_clients (END)#############")

    def train(self):
        for hfltrainer in self.federation_list:
            hfltrainer.train()



