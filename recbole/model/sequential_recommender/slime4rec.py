import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward
from recbole.model.loss import BPRLoss

"""
SLIME4Rec
################################################

Reference:
    Du, Xinyu, et al. "Contrastive enhanced slide filter mixer for sequential recommendation." 2023 IEEE 39th International Conference on Data Engineering (ICDE). IEEE, 2023.

Reference:
    https://github.com/sudaada/SLIME4Rec

"""

class FilterMixerLayer(nn.Module):
    def __init__(self, hidden_size, i, config):
        super(FilterMixerLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.config = config
        self.filter_mixer = config['filter_mixer']
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.complex_weight = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        if self.filter_mixer == 'G':
            self.complex_weight_G = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        elif self.filter_mixer == 'L':
            self.complex_weight_L = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        elif self.filter_mixer == 'M':
            self.complex_weight_G = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
            self.complex_weight_L = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)

        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.n_layers = config['n_layers']

        self.dynamic_ratio = config['dynamic_ratio']
        self.slide_step = ((self.max_item_list_length // 2 + 1) * (1 - self.dynamic_ratio)) // (self.n_layers - 1)

        self.static_ratio = 1 / self.n_layers
        self.filter_size = self.static_ratio * (self.max_item_list_length // 2 + 1)

        self.slide_mode = config['slide_mode']
        if self.slide_mode == 'one':
            G_i = i
            L_i = self.n_layers - 1 - i
        elif self.slide_mode == 'two':
            G_i = self.n_layers - 1 - i
            L_i = i
        elif self.slide_mode == 'three':
            G_i = self.n_layers - 1 - i
            L_i = self.n_layers - 1 - i
        elif self.slide_mode == 'four':
            G_i = i
            L_i = i
        # print("slide_mode:", self.slide_mode, len(self.slide_mode), type(self.slide_mode))


        if self.filter_mixer == 'G' or self.filter_mixer == 'M':
            self.w = self.dynamic_ratio
            self.s = self.slide_step
            if self.filter_mixer == 'M':
                self.G_left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (G_i * self.s))
                self.G_right = int((self.max_item_list_length // 2 + 1) - G_i * self.s)
                print("====================================================================================G_left, right",
                  self.G_left, self.G_right, self.G_right - self.G_left)
            self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (G_i * self.s))
            self.right = int((self.max_item_list_length // 2 + 1) - G_i * self.s)
            


        if self.filter_mixer == 'L' or self.filter_mixer == 'M':
            self.w = self.static_ratio
            self.s = self.filter_size
            if self.filter_mixer == 'M':
                self.L_left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (L_i * self.s))
                self.L_right = int((self.max_item_list_length // 2 + 1) - L_i * self.s)
                print("====================================================================================L_left, Light",
                  self.L_left, self.L_right, self.L_right - self.L_left)

            self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (L_i * self.s))
            self.right = int((self.max_item_list_length // 2 + 1) - L_i * self.s)
            
            

    def forward(self, input_tensor):
        # print("input_tensor", input_tensor.shape)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        if self.filter_mixer == 'M':
            weight_g = torch.view_as_complex(self.complex_weight_G)
            weight_l = torch.view_as_complex(self.complex_weight_L)
            G_x = x
            L_x = x.clone()
            G_x[:, :self.G_left, :] = 0
            G_x[:, self.G_right:, :] = 0
            output = G_x * weight_g

            L_x[:, :self.L_left, :] = 0
            L_x[:, self.L_right:, :] = 0
            output += L_x * weight_l


        else:
            weight = torch.view_as_complex(self.complex_weight)
            x[:, :self.left, :] = 0
            x[:, self.right:, :] = 0
            output = x * weight

        sequence_emb_fft = torch.fft.irfft(output, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)

        if self.config['residual']:
            origianl_out = self.LayerNorm(hidden_states + input_tensor)
        else:
            origianl_out = self.LayerNorm(hidden_states)

        return origianl_out

class FMBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_dropout_prob,
                 hidden_act,
                 layer_norm_eps,
                 i,
                 config,
                 ) -> None:
        super().__init__()
        self.intermediate = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps, config)
        self.filter_mixer_layer = FilterMixerLayer(hidden_size, i, config)
        
    def forward(self, x):
        out = self.filter_mixer_layer(x)
        out = self.intermediate(out)
        return out


class Encoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
            self,
            n_layers=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            inner_skip_type='straight',
            outer_skip_type='straight',
            simgcl_lambda=0,
            inner_wide=False,
            outer_wide=False,
            add_detach=False,
            fine_grained=26,
            learnable=False,
            config=None,
    ):

        super(Encoder, self).__init__()

        self.outer_skip_type = outer_skip_type
        self.simgcl_lambda = simgcl_lambda

        # self.attention_layer = MultiHeadAttention(config['n_heads'], hidden_size, hidden_dropout_prob, config['attn_dropout_prob'], layer_norm_eps)
        self.n_layers = config['n_layers']

        self.layer = nn.ModuleList()
        for n in range(self.n_layers):
            self.fmblock = FMBlock(
                hidden_size,
                inner_size,
                hidden_dropout_prob,
                hidden_act,
                layer_norm_eps,
                n,
                config)
            self.layer.append(self.fmblock)



    def forward(self, hidden_states, output_all_encoded_layers):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """

        all_encoder_layers = []
        for layer_module in self.layer:
            # if self.training:
            #     # simgcl
            #     # print("hidden_states:!!!!!!!!!!!!!!!!!!!!", layer_module)
            #     random_noise = torch.FloatTensor(hidden_states.shape).uniform_(0, 1).to('cuda')
            #     hidden_states += self.simgcl_lambda * torch.multiply(torch.sign(hidden_states),
            #                                                          torch.nn.functional.normalize(random_noise,
            #                                                                                        p=2, dim=1))
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # hidden_states = self.attention_layer(hidden_states, attention_mask)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers
    
class SLIME4Rec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SLIME4Rec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.lmd = config['lmd']
        self.lmd_sem = config['lmd_sem']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        # dxy
        self.inner_skip_type = config['inner_skip_type']
        self.outer_skip_type = config['outer_skip_type']
        self.inner_wide = config['inner_wide']
        self.outer_wide = config['outer_wide']
        self.add_detach = config['add_detach']
        self.fine_grained = config['fine_grained']
        self.learnable = config['learnable']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.my_encoder = Encoder(
        n_layers=self.n_layers,
        hidden_size=self.hidden_size,
        inner_size=self.inner_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        hidden_act=self.hidden_act,
        layer_norm_eps=self.layer_norm_eps,
        inner_skip_type=self.inner_skip_type,
        outer_skip_type=self.outer_skip_type,
        # simgcl_lambda=self.simgcl_lambda,
        inner_wide=self.inner_wide,
        outer_wide=self.outer_wide,
        add_detach=self.add_detach,
        fine_grained=self.fine_grained,
        learnable=self.learnable,
        config=config)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.ssl = config['contrast']
        self.tau = config['tau']
        self.sim = config['sim']
        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            # module.weight.data = self.truncated_normal_(tensor=module.weight.data, mean=0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
        

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        # extended_attention_mask = self.get_bi_attention_mask(item_seq)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!input_emb", input_emb)

        trm_output = self.my_encoder(input_emb, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        batch, seq_len = item_seq.shape
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # simgrace
        # seq_output_vice = self.gen_ran_output(item_seq, item_seq_len, vice_model)
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        if self.ssl == 'us_x':
            aug_seq_output = self.forward(item_seq, item_seq_len)
            sem_aug, sem_aug_lengths = interaction['sem_aug'], interaction['sem_aug_lengths']
            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)
            sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                           batch_size=item_seq_len.shape[0], sim=self.sim)
            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
        
        return loss
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores