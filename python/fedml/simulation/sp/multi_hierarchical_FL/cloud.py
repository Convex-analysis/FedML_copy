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
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
            

"""

"""
