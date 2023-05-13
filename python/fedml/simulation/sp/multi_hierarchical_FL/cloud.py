import itertools
import random
import time

import numpy
import logging
import numpy as np
import torch
import wandb
import copy
from fedml import mlops

from .group import Group
from ..hierarchical_fl.trainer import HierarchicalTrainer


def generate_array_with_constant_sum(n, total_sum):
    # initialize array with all 1's
    arr = [1] * n
    remaining_sum = total_sum - n  # subtract n since we have n elements already

    # distribute remaining sum randomly
    i = 0
    while remaining_sum > 0:
        arr[i] += 1
        remaining_sum -= 1
        i = (i + 1) % n

    return arr


class Cloud(HierarchicalTrainer):
    def __init__(self, idx, args, device, model, model_trainer, client_list):
        self.idx = idx
        self.args = args
        self.device = device
        self.model = model
        self.model_trainer = model_trainer
        self.group_list = None
        self.group_indexes = None
        self.client_list = client_list
        self.test_data_local_dict = None
        self.train_data_local_num_dict = None
        self.train_data_local_dict = None

        self.run_time = 0
        self.mantaince_cost = 0
        self.computation_cost = 0
        self.communication_cost = 0
        self.computation_unit_cost = random.uniform(0, 0.1)
        self.communication_unit_cost = random.uniform(0.05, 0.2)
        self.mantaince_unit_cost = random.uniform(0, 0.2)

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
        group_diff_dict = {}
        start_time = time.time()
        for global_round_idx in range(self.args.comm_round):
            logging.info(
                "################Global Communication Round : {}".format(
                    global_round_idx
                )
            )
            if self.args.group_participation_method == "uniform":
                group_to_client_indexes = self._all_group_participate_client_sampling(
                    global_round_idx,
                    self.group_list,
                    self.args.client_num_per_round,
                )
            elif self.args.group_participation_method == "random":
                group_to_client_indexes = self._client_sampling(
                global_round_idx,
                self.group_list,
                self.args.client_num_per_round,
                )
            else:
                Exception("Unknown group participation method {}".format(self.args.group_participation_method))
            # train each group
            w_groups_dict = {}
            train_order = []
            for group in self.group_list:

                sampled_client_indexes = [client for client in group_to_client_indexes if
                                          client in group.client_dict.keys()]
                if len(sampled_client_indexes) >= 1:
                    train_order.append(group.idx)
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

                """
                measure the difference from w_global and each group model and store the differ in a dictionary
                group_diff_dict = globalepoch : groups differ
                groups differ is a list :   [(groupid, diff)]
                """
                # evaluate performance
                if (
                        global_epoch == 0
                        or (global_epoch + 1) % self.args.frequency_of_the_test == 0
                        or global_epoch
                        == self.args.comm_round
                        * self.args.group_comm_round
                        * self.args.epochs
                        - 1
                ):
                    self.model.load_state_dict(w_global)
                    if not global_epoch in w_groups_dict:
                        group_diff_dict[global_epoch] = []
                    group_diff_dict[global_epoch] = self._get_model_diff(w_global, w_groups, train_order, global_epoch)
                    # self._local_test_on_all_clients(global_epoch)
                    self.test_on_local_clients(global_epoch)

            """
                        for global_epochs in sorted(group_diff_dict.keys()):
                w_groups_diff = group_diff_dict[global_epochs]
                for idx, diff in w_groups_diff:
                    for group in self.group_list:
                        if group.idx == idx:
                            group.diff += diff
            """
        end_time = time.time()
        self.run_time = end_time - start_time
        self._calculate_cost()
        #In this stage we caculated the diff of each group during training process
        for group in self.group_list:
            diff = 0
            spcfc_group_differ = []
            for gpochs in sorted(group_diff_dict.keys()):
                for idx, diff in group_diff_dict[gpochs]:
                    if idx == group.idx:
                        spcfc_group_differ.append(diff)
            for item in spcfc_group_differ:
                diff = diff * 0.7 + float(item)
            group.diff = diff
        #print([(group.idx, group.diff) for group in self.group_list])
        self.local_test_on_acc_diff()
        """
        #gengerate the combination of self.group_list
        group_combination = []
        for i in range(1, len(self.group_list) + 1):
            group_combination.extend(list(itertools.combinations(self.group_list, i)))
        """






    def _client_sampling(self, global_round_idx, group_list, client_num_per_round):
        # store all groups' client and ready to sample
        total_canadiate_client = []
        for group in group_list:
            total_canadiate_client.extend(group.client_dict.keys())
        if total_canadiate_client == client_num_per_round:
            client_indexes = [client_index for client_index in range(total_canadiate_client)]
        else:
            num_clients = min(client_num_per_round, len(total_canadiate_client))
            np.random.seed(
                global_round_idx)  # make sure for each comparison, we are selecting the same clients each round
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

    def _all_group_participate_client_sampling(self, global_round_idx, group_list, client_num_per_round):
        client_indexes = []
        group_to_client_indexes = {}
        #根据group的数量将client_num_per_round分配到每个group中
        selected_array = generate_array_with_constant_sum(len(group_list), client_num_per_round)
        for idx, group in enumerate(group_list):
            #print(group.client_dict.keys())
            np.random.seed(
                global_round_idx+idx)  # make sure for each comparison, we are selecting the same clients each round
            sampled_client_index = np.random.choice(list(group.client_dict.keys()), selected_array[idx], replace=False)
            if not group.idx in group_to_client_indexes:
                group_to_client_indexes[group.idx] = []
            group_to_client_indexes[group.idx].append(sampled_client_index)
            client_indexes.extend(sampled_client_index)
        logging.info("client_indexes = %s" % str(client_indexes))
        logging.info(
            "client_indexes of each group = {}".format(group_to_client_indexes)
        )
        return client_indexes


    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc"+self.idx : train_acc, "round": round_idx})
            wandb.log({"Train/Loss"+self.idx : train_loss, "round": round_idx})

        mlops.log({"Train/Acc"+self.idx : train_acc, "round": round_idx})
        mlops.log({"Train/Loss"+self.idx : train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc"+self.idx : test_acc, "round": round_idx})
            wandb.log({"Test/Loss"+self.idx : test_loss, "round": round_idx})

        mlops.log({"Test/Acc"+self.idx : test_acc, "round": round_idx})
        mlops.log({"Test/Loss"+self.idx : test_loss, "round": round_idx})
        logging.info(stats)

    def test_on_local_clients(self, round_idx):
        logging.info("################test_on_local_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        total_canadiate_client = []
        
        for group in self.group_list:
            total_canadiate_client.extend(group.client_dict.keys())

        for client_idx in total_canadiate_client:
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )

            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc" + " of Federation " + str(self.idx): train_acc, "round": round_idx})
            wandb.log({"Train/Loss" + " of Federation " + str(self.idx): train_loss, "round": round_idx})

        mlops.log({"Train/Acc" + " of Federation " + str(self.idx): train_acc, "round": round_idx})
        mlops.log({"Train/Loss" + " of Federation " + str(self.idx): train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc" + " of Federation " + str(self.idx): test_acc, "round": round_idx})
            wandb.log({"Test/Loss" + " of Federation " + str(self.idx): test_loss, "round": round_idx})

        mlops.log({"Test/Acc" + " of Federation " + str(self.idx): test_acc, "round": round_idx})
        mlops.log({"Test/Loss" + " of Federation " + str(self.idx): test_loss, "round": round_idx})
        logging.info(stats)


    def set_test_data_dict(self, test_data_local_dict):
        self.test_data_local_dict = test_data_local_dict

    def set_train_data_local_num_dict(self, train_data_local_num_dict):
        self.train_data_local_num_dict = train_data_local_num_dict

    def set_train_data_local_dict(self, train_data_local_dict):
        self.train_data_local_dict = train_data_local_dict

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        initial_params = copy.deepcopy(w_locals[0])

        (sample_num, averaged_params) = initial_params
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _get_model_diff(self, w_global, w_groups, train_order, round_idx):
        """
        measure the difference from w_global and each group model
        """
        diff_list = []
        for idx, w_group in zip(train_order, w_groups):
            diff = 0
            for k in w_global.keys():
                diff += torch.norm(w_global[k] - w_group[1][k])

            logging.info("######the gap between gloabl model and model of group {} diff = {}######".format(idx, diff))
            """
                        if self.args.enable_wandb:
                wandb.log({"Train/Diff in Federation " + str(self.idx) + " Group " + str(idx): diff, "round": round_idx})
            """
            diff_list.append((idx, diff))
        return diff_list

    def _calculate_cost(self):
        """
        calculate the cost of each group
        """
        total_cost = 0
        cost_list = []
        for group in self.group_list:
            cost = group.calculate_cost()
            cost_list.append((group.idx, cost))
        self.mantaince_cost = self.mantaince_unit_cost * self.run_time
        self.computation_cost = self.computation_unit_cost * self.args.comm_round
        self.communication_cost = self.communication_unit_cost * self.args.comm_round
        total_cost = self.mantaince_cost + self.computation_cost + self.communication_cost
        for cost in cost_list:
            total_cost += cost[1]
        logging.info("######the total cost of cloud {} is {}######".format(self.idx, total_cost))
        return cost_list

    def local_test_on_acc_diff(self):
        sample, diff = 0, 0
        for group in self.group_list:
            sample += group.get_sample_number()
            diff += group.diff * group.get_sample_number()
        avg_diff = diff/sample
        #avg_diff = sum([group.diff for group in self.group_list]) / len(self.group_list)
        avg_diff = int(avg_diff * 10000) / 10000.0
        avg_diff = avg_diff/len(self.group_list)

        logging.info("################test_on_local_clients : ")
        group_index = [group.idx for group in self.group_list]
        train_vs_diff = "Train/Acc of Gourps {}".format(group_index)
        test_vs_diff = "Test/Acc of Gourps {}".format(group_index)

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        total_canadiate_client = []

        for group in self.group_list:
            total_canadiate_client.extend(group.client_dict.keys())

        for client_idx in total_canadiate_client:
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )

            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))


        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])


        stats_train = {"training_acc": train_acc, "training_loss": train_loss, "diff": avg_diff, "group_index": group_index}
        logging.info(stats_train)

        return stats_train

