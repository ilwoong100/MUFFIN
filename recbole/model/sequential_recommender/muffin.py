
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.loss import EmbLoss
from recbole.model.abstract_recommender import SequentialRecommender
import copy
import math
from recbole.model.sequential_recommender.module import LFMEncoder, GFMEncoder
    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'silu':F.silu}


class Muffin(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"] 
        self.inner_size = config["inner_size"]  
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.lfm_encoder = LFMEncoder(config)
        self.gfm_encoder = GFMEncoder(config)
        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.concat_layer = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.kernel_size= config['kernel_size']

        # UAF
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size ,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

    
    def sequence_mask(self, input_ids):
        mask = (input_ids != 0) * 1
        return mask.unsqueeze(-1) 
    
    def make_embedding(self, sequence, seq_mask):
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = item_embeddings
        item_embeddings *= seq_mask
        item_embeddings = self.LayerNorm(item_embeddings)
        item_embeddings = self.dropout(item_embeddings)
        return item_embeddings
    
    def forward(self, input_ids, item_seq_len):
        seq_mask = self.sequence_mask(input_ids)
        sequence_emb = self.make_embedding(input_ids, seq_mask)

        # UAF
        frequency_emb = torch.fft.rfft(sequence_emb, dim=1,norm='ortho')
        filter = torch.sigmoid(self.freq_conv_encoder(frequency_emb.abs().permute(0,2,1)))
        
        # GFM
        gfm_layer = self.gfm_encoder(sequence_emb, seq_mask, filter,output_all_encoded_layers=True)
        gfm_output = gfm_layer[-1]
        gfm_output = self.gather_indexes(gfm_output, item_seq_len-1)

        # LFM
        item_encoded_layers, total_lb_loss = self.lfm_encoder(sequence_emb, seq_mask, filter, output_all_encoded_layers=True)
        lfm_output = item_encoded_layers[-1]
        lfm_output = self.gather_indexes(lfm_output, item_seq_len - 1)
        
        concate_output = torch.cat((lfm_output, gfm_output),dim=-1)
        output = self.concat_layer(concate_output)

        last_hidden_state = self.gather_indexes(sequence_emb, item_seq_len - 1)
        output = self.LayerNorm(output + last_hidden_state)
        output = self.dropout(output)
        return output, gfm_output, lfm_output, total_lb_loss

     
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output, gfm_output, lfm_output, total_lb_loss = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        
        # add auxiliary loss
        logits = torch.matmul(gfm_output, test_item_emb.transpose(0,1))
        gfm_loss = self.loss_fct(logits, pos_items)
        logits = torch.matmul(lfm_output, test_item_emb.transpose(0,1))
        lfm_loss = self.loss_fct(logits, pos_items) 
        loss = loss + self.alpha*(gfm_loss + lfm_loss)

        # add load balancing loss
        loss += self.beta * total_lb_loss
        return loss
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output,_ ,_,_ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embeddings.weight
        
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) 
        return scores

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
