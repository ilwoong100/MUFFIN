# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
FMLPRec
################################################

Reference:
    Zhou, Kun, et al. "Filter-enhanced MLP is all you need for sequential recommendation." Proceedings of the ACM web conference 2022. 2022.

Reference:
    https://github.com/Woeee/FMLP-Rec

"""
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward
from recbole.model.loss import BPRLoss
from IPython import embed

class FMLPEncoder(nn.Module):
    def __init__(
        self,
        config,
        n_layers=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(FMLPEncoder, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer = nn.ModuleList()
        for n in range(n_layers):
            self.fmblock = FMLPLayer(
                config,
                hidden_size,
                inner_size,
                hidden_dropout_prob,
                hidden_act,
                layer_norm_eps
            )
            self.layer.append(self.fmblock)

    def forward(self, hidden_states, output_all_encoded_layers=True):

        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers

class FMLPLayer(nn.Module):
    def __init__(
        self,
        config,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps
    ):
        super(FMLPLayer, self).__init__()
        self.filter_layer = FilterLayer(config)
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
            config
        )

    def forward(self, hidden_states):
        filter_output = self.filter_layer(hidden_states)
        feedforward_output = self.feed_forward(filter_output)
        return feedforward_output



class FilterLayer(nn.Module):
    def __init__(self, config):
        super(FilterLayer, self).__init__()
    
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)

        self.freq_weight = nn.Parameter(torch.randn(1, config['MAX_ITEM_LIST_LENGTH']//2 + 1, config['hidden_size'], 2, dtype=torch.float32) * 0.02)

    def forward(self, input_tensor):
        batch, max_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        
        freq_weight = torch.view_as_complex(self.freq_weight)
        use_emb = x * freq_weight    
        sequence_signal = torch.fft.irfft(use_emb, n=max_len, dim=1, norm='ortho') 
        hidden_states = self.out_dropout(sequence_signal)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FMLPRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(FMLPRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.freq_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]       
        
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = FMLPEncoder(
            config = config,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)


    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
            
        trm_output = self.trm_encoder(
                input_emb, output_all_encoded_layers=True
                )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  


        
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))     
            loss = self.loss_fct(logits, pos_items)

        return loss


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()