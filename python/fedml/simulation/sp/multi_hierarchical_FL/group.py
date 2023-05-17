import logging
import random
import time

import numpy.random

from .client import HFLClient
from ..fedavg.fedavg_api import FedAvgAPI


class Group(FedAvgAPI):
    def __init__(
            self,
            idx,
            total_client_indexes,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            args,
            device,
            model,
            model_trainer,
    ):
        self.idx = idx
        self.args = args
        self.device = device
        self.client_dict = {}
        self.train_data_local_num_dict = train_data_local_num_dict
        self.diff = 0
        self.w_group = None

        #enconamic cost
        random.seed(self.idx)
        self.run_time = 0
        self.mantaince_cost = 0
        self.computation_cost = 0
        self.communication_cost = 0
        self.computation_unit_cost = random.uniform(0, 0.1)
        self.communication_unit_cost = random.uniform(0.1, 0.3)
        self.mantaince_unit_cost = random.uniform(0, 0.1)
        self.total_cost = 0

        for client_idx in total_client_indexes:
            self.client_dict[client_idx] = HFLClient(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                args,
                device,
                model,
                model_trainer,
            )

    def get_sample_number(self, sampled_client_indexes):
        self.group_sample_number = 0
        for client_idx in sampled_client_indexes:
            self.group_sample_number += self.train_data_local_num_dict[client_idx]
        return self.group_sample_number

    def train(self, global_round_idx, w, sampled_client_indexes):
        sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes]
        w_group = w
        w_group_list = []
        start_timestamp = time.time()
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # train each client
            for client in sampled_client_list:
                w_local_list = client.train(global_round_idx, group_round_idx, w_group)
                for global_epoch, w in w_local_list:
                    if not global_epoch in w_locals_dict:
                        w_locals_dict[global_epoch] = []
                    w_locals_dict[global_epoch].append((client.get_sample_number(), w))

            # aggregate local weights
            for global_epoch in sorted(w_locals_dict.keys()):
                w_locals = w_locals_dict[global_epoch]
                w_group_list.append((global_epoch, self._aggregate(w_locals)))
            if w_group_list == []:
                pass
            # update the group weight
            w_group = w_group_list[-1][1]
        self.w_group = w_group
        end_timestamp = time.time()
        random.seed(self.idx)
        self.set_run_time(self.args.group_comm_round * random.uniform(0.1, 0.3))
        self.set_computation_cost(self.computation_unit_cost * self.args.group_comm_round)
        self.set_communication_cost(self.communication_unit_cost * self.args.group_comm_round)
        return w_group_list

    def calculate_cost(self):
        self.total_cost = self.mantaince_cost + self.computation_cost + self.communication_cost
        return self.total_cost
    def set_run_time(self, run_time):
        mantaince_cost = random.randint(1, 10)
        self.run_time = run_time
        self.mantaince_cost = mantaince_cost * run_time

    def set_computation_cost(self, param):
        self.computation_cost = param

    def set_communication_cost(self, param):
        self.communication_cost = param
