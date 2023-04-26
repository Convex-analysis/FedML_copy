from ..hierarchical_fl.trainer import HierarchicalTrainer


class Cloud(HierarchicalTrainer):
    def _client_sampling(
            self, global_round_idx, client_num_in_total, client_num_per_round
    ):
        sampled_client_indexes = super()._client_sampling(
            global_round_idx, client_num_in_total, client_num_per_round
        )
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)

        sampled_group_indexes = []
        for group_idx in group_to_client_indexes.keys():
            sampled_group_indexes.append(group_idx)

        return sampled_group_indexes