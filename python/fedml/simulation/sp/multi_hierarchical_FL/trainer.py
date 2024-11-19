import os.path
import time

import numpy
import copy
import logging
import json

import torch
import numpy as np
import pandas as pd
import wandb

from .client import HFLClient
from .cloud import Cloud
from .group import Group

from fedml.ml.trainer.trainer_creator import create_model_trainer


# 解析json文件，返回一个字典
def parse_json_file(json_file):
    root = os.getcwd()

    with open(os.path.join(root, json_file), "r") as f:
        data = json.load(f)
    return data

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not np.array_equal(p1.cpu().numpy(), p2.cpu().numpy()):
            return False
    return True

def get_model_diff(model1, model2):
        """
        measure the difference from w_global and each group model
        """
        diff = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()): 
            diff += torch.sum(torch.abs(p1 - p2)).item() 
        return diff


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
        # self.model_trainer = None
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
        if self.args.group_partition_type == "random":
            self._setup_groups_for_MHFL()
        elif self.args.group_partition_type == "constant":
            self._setup_groups_for_MHFL_from_file()
        elif self.args.group_partition_type == "profile":
            self._setup_groups_for_MHFL_from_profile()

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
        np.random.seed(len(self.federation_list))
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

    def _setup_groups_for_MHFL_from_file(self):
        logging.info("############_setup_groups_for_MHFL_from_file (START)#############")
        cloud_to_group = parse_json_file(self.args.group_partition_file)
        for federation in cloud_to_group.items():
            group_list = []
            idx = federation[1]["id"]
            tmp_group_list = federation[1]["member"]
            if idx <= len(self.federation_list) - 1:
                cloud = self.federation_list[idx]
                for group_idx in tmp_group_list:
                    group_list.append(self.group_dict[group_idx])
                cloud.set_group_list(group_list)
                cloud.set_group_indexes(group_list)
        logging.info("############_setup_groups_for_MHFL_from_file (END)#############")
    
    #只适应于1个federation
    def _setup_groups_for_MHFL_from_profile(self):
        logging.info("############_setup_groups_for_MHFL_from_profile (START)#############")
        tmp_group_list = self.args.group_partition_profile
        group_list = []
        if len(self.federation_list) == 1:
            cloud = self.federation_list[0]
            for group_idx in tmp_group_list:
                group_list.append(self.group_dict[group_idx])
            cloud.set_group_list(group_list)
            cloud.set_group_indexes(group_list)
        logging.info("############_setup_groups_for_MHFL_from_profile (END)#############")

    # setting up federations in our simulation
    def _setup_federations(self, federation_num):
        logging.info("############setup_federations_for_MHFL (START)#############")
        list_of_federation = []
        for i in range(federation_num):
            # test_client_list = copy.deepcopy(self.client_list)
            test_client_list = self.client_list
            model = copy.deepcopy(self.model)
            # model = self.model
            # model_trainer = copy.deepcopy(self.model_trainer)
            model_trainer = self.model_trainer
            list_of_federation.append(Cloud(i, self.args, self.device, model, model_trainer, test_client_list))
        for hfl in list_of_federation:
            hfl.set_test_data_dict(self.test_data_local_dict)
            hfl.set_train_data_local_num_dict(self.train_data_local_num_dict)
            hfl.set_train_data_local_dict(self.train_data_local_dict)

        self.federation_list = list_of_federation
        logging.info("############setup_federations_for_MHFL (END)#############")

    def train(self):
        tmp_cloud = None
        flag = False
        cloud_train_stats_list = []
        logging.info("############Multi Federation Training (START)#############")
        for hfl in self.federation_list:
            logging.info("############Training Cloud {} (START)#############".format(hfl.idx))
            cloud_train_stats_list.append(hfl.train())
            '''
            if tmp_cloud != None:
                flag = compare_models(tmp_cloud.model,hfl.model)
            if flag:
                logging.info("Cloud {} and Cloud {} have same parameters".format(hfl.idx, tmp_cloud.idx)) 
            '''
            if tmp_cloud != None:
                tmp_difference = get_model_diff(tmp_cloud.model,hfl.model)
                logging.info("Cloud {} and Cloud {} model differences is {}".format(hfl.idx, tmp_cloud.idx, tmp_difference)) 
            tmp_cloud = hfl
            
        logging.info("############Multi Federation Training (END)#############")
        # self.test_diff_vs_acc()
        logging.info("############Multi Federation Testing (START)#############")
        for tmp_stats in cloud_train_stats_list:
            if self.args.enable_wandb:
                trainacc = tmp_stats["training_acc"]
                diff = tmp_stats["diff"]
                group_index = tmp_stats["group_index"]
                train_vs_diff = "Train/Acc of Gourps {}".format(group_index)
                wandb.log({"Train/Acc": trainacc, "model_diff": diff})
        # 将cloud_train_stats_list转化为dataframe
        df = pd.DataFrame(cloud_train_stats_list)
        # 将dataframe写入csv文件
        # 将本地时间作为文件名
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        log_result_csv = str(self.args.federation_num) + "federations-" + str(self.args.dataset) + "-" + str(
            self.args.method) + "-" + str(self.args.group_participation_method) + "-" + str(localtime) + "-result.csv"
        df.to_csv(log_result_csv, index=False)

        """
                for i in range(3):
            tmp_cloud.append(self.federation_list[i])
        for i in range(len(self.federation_list)):
            if self.federation_list[i] in tmp_cloud:
                pass
            else:
                #get model parameters
                model_params = self.federation_list[i].model.state_dict()
                for cloud in tmp_cloud:
                    cloud_model_params = cloud.model.state_dict()
                    #calculate the difference between the model parameters
                    diff = 0
                    for key in model_params.keys():
                        diff += torch.norm(model_params[key] - cloud_model_params[key])
                    logging.info("diff between cloud {} and cloud {} is {}".format(i, cloud.idx, diff))
        """
        logging.info("############Multi Federation Testing (END)#############")

    def test_diff_vs_acc(self):
        tmp_stats = {}
        group_index = []
        logging.info("############Multi Federation Testing (START)#############")
        for hfl in self.federation_list:
            tmp_stats = hfl.local_test_on_acc_diff()
            if self.args.enable_wandb:
                trainacc = tmp_stats["training_acc"]
                diff = tmp_stats["diff"]
                group_index = tmp_stats["group_index"]
                train_vs_diff = "Train/Acc of Gourps {}".format(group_index)
                wandb.log({"Train/Acc": trainacc, "model_diff": diff})
        # 遍历federation_list的第一到三个元素
        for i in range(3):
            tmp_stats.append(self.federation_list[i])
        for i in range(len(self.federation_list)):
            if self.federation_list[i] in tmp_stats:
                pass
            else:
                # get model parameters
                model_params = self.federation_list[i].model.state_dict()
                for cloud in tmp_stats:
                    cloud_model_params = cloud.model.state_dict()
                    # calculate the difference between the model parameters
                    diff = 0
                    for key in model_params.keys():
                        diff += torch.sum(torch.abs(model_params[key] - cloud_model_params[key]))
                    diff = diff / len(model_params.keys())
                    logging.info("diff between cloud {} and cloud {} is {}".format(i, cloud.idx, diff))

        logging.info("############Multi Federation Testing (END)#############")
