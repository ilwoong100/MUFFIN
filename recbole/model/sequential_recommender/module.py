import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
    
    
    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'silu':F.silu}


class LFMGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_bands = config['num_bands']
        
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),  
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.num_bands)
        )

    def forward(self, x):
        magnitude = x.abs()
        phase = torch.angle(x)
        mag_features = torch.mean(magnitude, dim=1)
        phase_features = torch.mean(phase, dim=1)
        combined_features = torch.cat([mag_features, phase_features], dim=-1)
        
        gate_logits = self.gate(combined_features)
        probs = F.softmax(gate_logits, dim=-1)
        
        local_band_prob, prob_indices = torch.topk(probs, self.num_bands, dim=-1)
        local_band_prob_normalized = local_band_prob / local_band_prob.sum(dim=-1, keepdim=True)
        
        return local_band_prob_normalized, prob_indices
    
    
class LFMfilterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.complex_weight1 = nn.Parameter(torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH']//2 + 1, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.conv_layers = config['conv_layers']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.num_bands = config['num_bands']
        self.LFMgate = LFMGate(config)
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.LFMgate = LFMGate(config)

    def compute_balance_loss(self, local_band_indices, local_band_prob):
        batch_size = local_band_indices.size(0)
        mask = F.one_hot(local_band_indices, num_classes=self.num_bands).float()
        weighted_mask = mask * local_band_prob.unsqueeze(-1)
        band_usage = weighted_mask.sum(dim=[0, 1])
        band_usage = band_usage / batch_size
        ideal_usage = torch.ones_like(band_usage) * (1 / self.num_bands)
        usage_penalty = (band_usage - ideal_usage) ** 2
        balance_loss =  usage_penalty.mean()
        return balance_loss, band_usage
    
    def forward(self, input_tensor, seq_mask,filter):
        batch, max_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        local_band_prob,  prob_indices = self.LFMgate(x)
        balance_loss, band_usage = self.compute_balance_loss(prob_indices, local_band_prob)
        weight = torch.view_as_complex(self.complex_weight1)
        
        filtered_weight = torch.complex(filter * weight.real , filter* weight.imag)
        x_ = x * filtered_weight.permute(0,2,1)

        frequency_bands = torch.empty((batch, self.num_bands, max_len, hidden), device=input_tensor.device, dtype=input_tensor.dtype)
        for band in range(self.num_bands):
            frequency_output = torch.zeros_like(x_)
            band_start = band * (max_len//2+1) // self.num_bands
            band_end = (band + 1) * (max_len//2+1) // self.num_bands
            frequency_output[:,band_start:band_end] = x_[:,band_start:band_end]  
            sequence_emb_fft = torch.fft.irfft(frequency_output, n=max_len, dim=1, norm='ortho')
            
            band_output = self.out_dropout(sequence_emb_fft)
            frequency_bands[:,band] = self.LayerNorm(band_output + input_tensor)
            
        selected = torch.gather(frequency_bands, dim=1, index=prob_indices.view(batch, self.num_bands, 1, 1).expand(-1, -1, max_len, hidden))
        weighted_bands = local_band_prob.view(batch, self.num_bands, 1, 1) * selected
        LFM_output = weighted_bands.sum(dim=1)
        
        return LFM_output, balance_loss
    

class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['inner_size'])
        if isinstance(config['hidden_act'], str):
            self.intermediate_act_fn = ACT2FN[config['hidden_act']]
        else:
            self.intermediate_act_fn = config['hidden_act']

        self.dense_2 = nn.Linear(config['inner_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filterlayer = LFMfilterLayer(config)
        self.intermediate = Intermediate(config)
    def forward(self, hidden_states, seq_mask, filter):
        LFM_output, balance_loss = self.filterlayer(hidden_states, seq_mask, filter)
        output = self.intermediate(LFM_output)
        return output, balance_loss


class LFMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = LFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['n_layers'])])

    def forward(self, hidden_states, seq_mask, filter ,output_all_encoded_layers=True):
        all_encoder_layers = []
        total_balance_loss = 0
        
        for layer_module in self.layer:
            hidden_states, balance_loss= layer_module(hidden_states, seq_mask, filter)
            total_balance_loss += balance_loss
            
            if output_all_encoded_layers:
                all_encoder_layers.append((hidden_states))
                
        if not output_all_encoded_layers:
            all_encoder_layers.append((hidden_states))
            
        return all_encoder_layers, total_balance_loss
    
    
class GFMFilterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.complex_weight1 = nn.Parameter(torch.randn(1, config['hidden_size'], config['MAX_ITEM_LIST_LENGTH']//2 + 1, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config['freq_dropout_prob'])
        self.conv_layers = config['conv_layers']
        self.hidden_size = config['hidden_size']
        self.kernel_size = config['kernel_size']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.num_bands = config['num_bands']
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )


    def forward(self, input_tensor, seq_mask, filter):
        batch, max_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight1)
            
            
        filtered_weight = torch.complex(filter * weight.real , filter* weight.imag )
        x_ = x * filtered_weight.permute(0,2,1)
        
        whole_sequence_emb_irfft = torch.fft.irfft(x_, n=max_len, dim=1, norm='ortho')
        whole_emb = self.out_dropout(whole_sequence_emb_irfft)
        whole_emb = self.LayerNorm(whole_emb + input_tensor)
        return whole_emb


class GFMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filterlayer = GFMFilterLayer(config)
        self.intermediate = Intermediate(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
            
    def forward(self, hidden_states, seq_mask, filter):
        gfm_output = self.filterlayer(hidden_states, seq_mask, filter)
        output = self.intermediate(gfm_output)
        return output


class GFMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = GFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['n_layers'])])

    def forward(self, hidden_states, seq_mask, filter, output_all_encoded_layers=True):
        all_encoder_layers = []        
        for layer_module in self.layer:
            hidden_states= layer_module(hidden_states, seq_mask, filter)            
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers
