import logging
import random
import time

import numpy.random
from .ClientSelection import config
from .ClientSelection.clustered import ClusteredSampling2
from .ClientSelection.divfl import DivFL
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
        self.selection_method = None
        kwargs = {'total': args.client_num_in_total, 'device': device}
        if args.method == 'DivFL':
            assert args.subset_ratio is not None
            self.selection_method = DivFL(**kwargs, subset_ratio=args.subset_ratio)
        elif args.method == 'Cluster2':
            self.selection_method = ClusteredSampling2(**kwargs, dist='L1')

        if args.method in config.NEED_SETUP_METHOD:
            self.selection_method.setup(train_data_local_dict)

        self.idx = idx
        self.args = args
        self.device = device
        self.client_dict = {}
        self.train_data_local_num_dict = train_data_local_num_dict
        self.diff = 0
        self.w_group = model

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
        selected_client_indices = []
        sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes]
        #如果设置了客户端选择，就取消之前的随机选择，将客户端list初始化为所有客户端
        if self.selection_method is not None:
            #let self.client_dict to a list
            sampled_client_list = list(self.client_dict.values())

        w_group = w
        w_group_list = []
        start_timestamp = time.time()
        for group_round_idx in range(self.args.group_comm_round):
            #logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}
            #TODO client selecttion method
            ###
            # sampled_client_indexes is the client ids of this group
            if self.args.method in config.NEED_INIT_METHOD:
                local_models = [client.get_model() for client in sampled_client_list]
                self.selection_method.init(self.w_group, local_models)
                del local_models



            if self.args.method in config.PRE_SELECTION_METHOD:
                sampled_client_list = self.selection_method.select(self.args.client_num_per_round, sampled_client_list, None)

            # train each client
            for client in sampled_client_list:
                w_local_list = client.train(global_round_idx, group_round_idx, w_group)
                for global_epoch, w in w_local_list:
                    if not global_epoch in w_locals_dict:
                        w_locals_dict[global_epoch] = []
                    w_locals_dict[global_epoch].append((client.get_sample_number(), w))
            client_indices = [client.client_idx for client in sampled_client_list]

            #训练后利用model选择
            if self.args.method not in config.PRE_SELECTION_METHOD:
                #print(f'> post-client selection {self.num_clients_per_round}/{len(client_indices)}')
                kwargs = {'n': len(sampled_client_indexes), 'client_idxs': client_indices, 'round': group_round_idx}
                # select by local models(gradients)
                if self.args.method in config.NEED_LOCAL_MODELS_METHOD:
                    local_models = [self.client_dict[idx].get_model() for idx in client_indices]
                    selected_client_indices = self.selection_method.select(**kwargs, metric=local_models)
                    del local_models
                #print(selected_client_indices)


            # aggregate local weights
            for global_epoch in sorted(w_locals_dict.keys()):
                w_locals = w_locals_dict[global_epoch]
                w_group_list.append((global_epoch, self._aggregate(w_locals)))
            if w_group_list == []:
                pass
            # update the group weight
            if self.selection_method is not None:
                # 便利所有的client，如果client的id在selected_client_indices中，就将其加入到w_locals中
                w_locals = []
                for id in selected_client_indices:
                    client = sampled_client_list[id]
                    w_locals.append((client.get_sample_number(),
                                        client.model.state_dict()))
                print(len(w_locals))
                w_group = self._aggregate(w_locals)
            else:
                w_group = w_group_list[-1][1]
        self.w_group.load_state_dict(w_group)
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
