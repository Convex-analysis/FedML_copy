import numpy
import copy
import logging

import numpy as np

from .client import HFLClient
from .cloud import Cloud
from .group import Group

from fedml.ml.trainer.trainer_creator import create_model_trainer


# 利用这个traininer初始化所有的HFLtrainer，分配客户端和数据
class MultiHierFLTrainer():
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        #self.model_trainer = None
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self.group_dict = None
        self.group_indexes = None

        self._setup_clients_for_MHFL(
            self.train_data_local_num_dict,
            self.train_data_local_dict,
            self.test_data_local_dict,
        )
        self._setup_federations(self.args.federation_num)
        self._setup_groups_for_MHFL()
        
    # resign all client to corresponding groups
    def _setup_clients_for_MHFL(
            self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict
    ):
        logging.info("############setup_clients_for_MHFL (START)#############")
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
        logging.info("############setup_clients_for_MHFL (END)#############")

    def _setup_groups_for_MHFL(self):
        logging.info("############setup_groups_for_MHFL (START)#############")
        # reassign groups to the corresponding clouds
        group_index_list = list(self.group_dict.keys())
        cloud_to_group = {}
        # shuffle the group index
        np.random.shuffle(group_index_list)
        total_cloud_number = len(self.federation_list)
        # reassign group indexes
        tmp_group_index_list = []
        for list_idx, group_index in enumerate(group_index_list):
            cloud_index = list_idx % total_cloud_number
            if not cloud_index in cloud_to_group:
                cloud_to_group[cloud_index] = []
            cloud_to_group[cloud_index].append(group_index)
        # reassign group objective
        for cloud_idx, cloud_list in cloud_to_group.items():
            group_temp_list = []
            for group_idx in cloud_list:
                group_temp_list.append(self.group_dict[group_idx])
            tmp_cloud = self.federation_list[cloud_idx]
            tmp_cloud.set_group_list(group_temp_list)
            tmp_cloud.set_group_indexes(group_temp_list)
        logging.info("############setup_groups_for_MHFL (END)#############")

    # setting up federations in our simulation
    def _setup_federations(self, federation_num):
        logging.info("############setup_federations_for_MHFL (START)#############")
        list_of_federation = []
        for i in range(federation_num):
            #test_client_list = copy.deepcopy(self.client_list)
            test_client_list = self.client_list
            model = copy.deepcopy(self.model)
            #model = self.model
            #model_trainer = copy.deepcopy(self.model_trainer)
            model_trainer = self.model_trainer
            list_of_federation.append(Cloud(i, self.args,  self.device, model, model_trainer, test_client_list))
        for hfl in list_of_federation:
            hfl.set_test_data_dict(self.test_data_local_dict)
            hfl.set_train_data_local_num_dict(self.train_data_local_num_dict)
            hfl.set_train_data_local_dict(self.train_data_local_dict)

        self.federation_list = list_of_federation
        logging.info("############setup_federations_for_MHFL (END)#############")

    def train(self):
        logging.info("############Multi Federation Training (START)#############")
        for hfl in self.federation_list:
            logging.info("############Training Cloud {} (START)#############".format(hfl.idx))
            hfl.train()
        logging.info("############Multi Federation Training (END)#############")