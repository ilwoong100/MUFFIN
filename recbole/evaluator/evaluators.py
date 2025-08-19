# -*- encoding: utf-8 -*-
# @Time    :   2020/08/04
# @Author  :   Kaiyuan Li
# @email   :   tsotfsk@outlook.com

# UPDATE
# @Time    :   2021/01/07, 2020/08/11, 2020/12/18
# @Author  :   Kaiyuan Li, Yupeng Hou, Zhichao Feng
# @email   :   tsotfsk@outlook.com, houyupeng@ruc.edu.cn, fzcbupt@gmail.com

"""
recbole.evaluator.evaluators
#####################################
"""

from collections import ChainMap
import copy
import numpy as np
import torch

from recbole.evaluator.abstract_evaluator import GroupedEvaluator, IndividualEvaluator
from recbole.evaluator.metrics import metrics_dict

# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP']}
# These metrics are typical in loss recommendations
loss_metrics = {metric.lower(): metric for metric in ['AUC', 'RMSE', 'MAE', 'LOGLOSS']}
# For GAUC
rank_metrics = {metric.lower(): metric for metric in ['GAUC']}

# group-based metrics
group_metrics = ChainMap(topk_metrics, rank_metrics)
# not group-based metrics
individual_metrics = ChainMap(loss_metrics)


class TopKEvaluator(GroupedEvaluator):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
       contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
       The metrics used calculate group-based metrics which considers the metrics scores averaged
       across users. Some of them are also limited to k.

    """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)

        self.topk = config['topk']
        self._check_args()

    def collect(self, interaction, scores_tensor):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`

        Returns:
            torch.Tensor : a matrix contain topk matrix and shape matrix

       """
        user_len_list = interaction.user_len_list

        scores_matrix = self.get_score_matrix(scores_tensor, user_len_list)
        scores_matrix = torch.flip(scores_matrix, dims=[-1])
        shape_matrix = torch.full((len(user_len_list), 1), scores_matrix.shape[1], device=scores_matrix.device)

        # get topk
        _, topk_idx = torch.topk(scores_matrix, max(self.topk), dim=-1)  # n_users x k

        # pack top_idx and shape_matrix
        result = torch.cat((topk_idx, shape_matrix), dim=1)
        return result

    def evaluate(self, batch_matrix_list, eval_data):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_len_list = eval_data.get_pos_len_list()
        batch_result = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        # unpack top_idx and shape_matrix
        topk_idx = batch_result[:, :-1]
        shapes = batch_result[:, -1]

        assert len(pos_len_list) == len(topk_idx)
        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, topk_idx, shapes)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(float(value[k - 1]), self.precision)

        return metric_dict

    def _check_args(self):

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, '
                        'but get `{}`'.format(topk)
                    )
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_idx, shapes):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (numpy.ndarray): a list of users' positive items
            topk_idx (numpy.ndarray): a matrix which contains the index of the topk items for users
            shapes (numpy.ndarray): a list which contains the columns of the padded batch matrix

        Returns:
            numpy.ndarray: a matrix which contains the metrics result

        """
        pos_idx_matrix = (topk_idx >= (shapes - pos_len_list).reshape(-1, 1))
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(pos_idx_matrix, pos_len_list)
            result_list.append(result)  # n_users x len(metrics) x len(ranks)
        result = np.stack(result_list, axis=0).mean(axis=1)  # len(metrics) x len(ranks)
        return result

    def __str__(self):
        msg = 'The TopK Evaluator Info:\n' + \
              '\tMetrics:[' + \
              ', '.join([topk_metrics[metric.lower()] for metric in self.metrics]) + \
              '], TopK:[' + \
              ', '.join(map(str, self.topk)) + \
              ']'
        return msg


class RankEvaluator(GroupedEvaluator):
    r"""Rank Evaluator is mainly used in ranking tasks except for topk tasks. Now, we support one
    rank metric containing `'GAUC'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users except for top-k metrics.

    """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)
        pass

    def get_user_pos_len_list(self, interaction, scores_tensor):
        """get number of positive items and all items in test set of each user

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`

        Returns:
            list: number of positive items,
            list: number of all items
        """
        pos_len_list = torch.Tensor(interaction.pos_len_list).to(scores_tensor.device)
        user_len_list = interaction.user_len_list
        return pos_len_list, user_len_list

    def average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        device = scores.device
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=device)

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = torch.arange(0, length, device=device).repeat(width).reshape(width, -1). \
            transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = .5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def collect(self, interaction, scores_tensor):
        """collect the rank intermediate result of one batch, this function mainly implements ranking
        and calculating the sum of rank for positive items. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`

        """
        pos_len_list, user_len_list = self.get_user_pos_len_list(interaction, scores_tensor)
        scores_matrix = self.get_score_matrix(scores_tensor, user_len_list)
        desc_scores, desc_index = torch.sort(scores_matrix, dim=-1, descending=True)

        # get the index of positive items in the ranking list
        pos_index = (desc_index < pos_len_list.reshape(-1, 1))

        avg_rank = self.average_rank(desc_scores)
        pos_rank_sum = torch.where(pos_index, avg_rank, torch.zeros_like(avg_rank)).sum(axis=-1).reshape(-1, 1)

        return pos_rank_sum

    def evaluate(self, batch_matrix_list, eval_data):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data

        Returns:
            dict: such as ``{'GAUC': 0.9286}``

        """
        pos_len_list = eval_data.get_pos_len_list()
        user_len_list = eval_data.get_user_len_list()
        pos_rank_sum = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        assert len(pos_len_list) == len(pos_rank_sum)

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(user_len_list, pos_len_list, pos_rank_sum)
        for metric, value in zip(self.metrics, result_list):
            key = '{}'.format(metric)
            metric_dict[key] = round(value, self.precision)

        return metric_dict

    def _calculate_metrics(self, user_len_list, pos_len_list, pos_rank_sum):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (numpy.ndarray): a list of users' positive items
            topk_idx (numpy.ndarray): a matrix which contains the index of the topk items for users

        Returns:
            numpy.ndarray: a matrix which contains the metrics result

        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(user_len_list, pos_len_list, pos_rank_sum)
            result_list.append(result)
        return result_list

    def __str__(self):
        msg = 'The Rank Evaluator Info:\n' + \
              '\tMetrics:[' + \
              ', '.join([rank_metrics[metric.lower()] for metric in self.metrics]) + \
              ']'
        return msg


class LossEvaluator(IndividualEvaluator):
    r"""Loss Evaluator is mainly used in rating prediction and click through rate prediction. Now, we support four
    loss metrics which contain `'AUC', 'RMSE', 'MAE', 'LOGLOSS'`.

    Note:
        The metrics used do not calculate group-based metrics which considers the metrics scores averaged
        across users. They are also not limited to k. Instead, they calculate the scores on the entire
        prediction results regardless the users.

    """

    def __init__(self, config, metrics):
        super().__init__(config, metrics)

        self.label_field = config['LABEL_FIELD']

    def collect(self, interaction, pred_scores):
        """collect the loss intermediate result of one batch, this function mainly
        implements concatenating preds and trues. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            pred_scores (tensor): the tensor of model output with a size of `(N, )`

        Returns:
            tensor : a batch of scores with a size of `(N, 2)`

        """
        true_scores = interaction[self.label_field].to(pred_scores.device)
        assert len(true_scores) == len(pred_scores)
        return self.get_score_matrix(true_scores, pred_scores)

    def evaluate(self, batch_matrix_list, *args):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches

        Returns:
            dict: such as {'AUC': 0.83}

        """
        concat = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        trues = concat[:, 0]
        preds = concat[:, 1]

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(trues, preds)
        for metric, value in zip(self.metrics, result_list):
            key = '{}'.format(metric)
            metric_dict[key] = round(value, self.precision)
        return metric_dict

    def _calculate_metrics(self, trues, preds):
        """get metrics result

        Args:
            trues (numpy.ndarray): the true scores' list
            preds (numpy.ndarray): the predict scores' list

        Returns:
            list: a list of metrics result

        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(trues, preds)
            result_list.append(result)
        return result_list

    def __str__(self):
        msg = 'The Loss Evaluator Info:\n' + \
              '\tMetrics:[' + \
              ', '.join([loss_metrics[metric.lower()] for metric in self.metrics]) + \
              ']'
        return msg


metric_eval_bind = [(topk_metrics, TopKEvaluator), (loss_metrics, LossEvaluator), (rank_metrics, RankEvaluator)]



class Collector(object):
    """The collector is used to collect the resource for evaluator.
    As the evaluation metrics are various, the needed resource not only contain the recommended result
    but also other resource from data and model. They all can be collected by the collector during the training
    and evaluation process.

    This class is only used in Trainer.

    """

    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = "full" in config["eval_args"]["mode"]
        self.topk = self.config["topk"]
        self.device = self.config["device"]

    def data_collect(self, train_data):
        """Collect the evaluation resource from training data.
        Args:
            train_data (AbstractDataLoader): the training dataloader which contains the training data.

        """
        if self.register.need("data.num_items"):
            item_id = self.config["ITEM_ID_FIELD"]
            self.data_struct.set("data.num_items", train_data.dataset.num(item_id))
        if self.register.need("data.num_users"):
            user_id = self.config["USER_ID_FIELD"]
            self.data_struct.set("data.num_users", train_data.dataset.num(user_id))
        if self.register.need("data.count_items"):
            self.data_struct.set("data.count_items", train_data.dataset.item_counter)
        if self.register.need("data.count_users"):
            self.data_struct.set("data.count_users", train_data.dataset.user_counter)

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.

        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`

        Returns:
            torch.Tensor: average_rank

        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])

        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352

        """
        length, width = scores.shape
        true_tensor = torch.full(
            (length, 1), True, dtype=torch.bool, device=self.device
        )

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = (
            torch.arange(0, length, device=self.device)
            .repeat(width)
            .reshape(width, -1)
            .transpose(1, 0)
            .reshape(-1)
        )
        dense = obs.view(-1).cumsum(0) + bias

        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = 0.5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self,
        scores_tensor: torch.Tensor,
        interaction,
        positive_u: torch.Tensor,
        positive_i: torch.Tensor,
    ):
        """Collect the evaluation resource from batched eval data and batched model output.
        Args:
            scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
            interaction(Interaction): batched eval data.
            positive_u(Torch.Tensor): the row index of positive items for each user.
            positive_i(Torch.Tensor): the positive item id for each user.
        """
        if self.register.need("rec.items"):
            # get topk
            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k
            self.data_struct.update_tensor("rec.items", topk_idx)

        if self.register.need("rec.topk"):
            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.topk", result)

        if self.register.need("rec.meanrank"):
            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)

            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(
                pos_index == 1, avg_rank, torch.zeros_like(avg_rank)
            ).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.meanrank", result)

        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", scores_tensor)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor(
                "data.label", interaction[self.label_field].to(self.device)
            )

    def model_collect(self, model: torch.nn.Module):
        """Collect the evaluation resource from model.
        Args:
            model (nn.Module): the trained recommendation model.
        """
        pass
        # TODO:

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """Collect the evaluation resource from total output and label.
        It was designed for those models that can not predict with batch.
        Args:
            eval_pred (torch.Tensor): the output score tensor of model.
            data_label (torch.Tensor): the label tensor.
        """
        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", eval_pred)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            self.data_struct.update_tensor("data.label", data_label.to(self.device))

    def get_data_struct(self):
        """Get all the evaluation resource that been collected.
        And reset some of outdated resource.
        """
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct