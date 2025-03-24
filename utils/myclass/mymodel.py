

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..iTransformer.Transformer_EncDec import Encoder, EncoderLayer
from ..iTransformer.SelfAttention_Family import FullAttention, AttentionLayer
from ..iTransformer.Embed import DataEmbedding_inverted




class SignalReconstructionModel(nn.Module):
    def __init__(self, configs, num_paths):
        super(SignalReconstructionModel, self).__init__()
        d_model = max([config.d_model for config in configs])
        self.paths = nn.ModuleList([PathComponents(config,d_model) for config in configs])
        # Assuming each path could potentially contribute a different sequence length, 
        # but here we initialize projector without a fixed output size
        # self.dynamic_projector = None

    def encode(self, x_enc, x_mark_enc, path_components):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) )
        x_enc /= stdev
        _, _, N = x_enc.shape
        enc_out = path_components.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = path_components.encoder(enc_out, attn_mask=None)
        return enc_out, means, stdev, N

    def decode(self, concatenated_enc_out, means, stdev, N, path_components):
    
        # self.dynamic_projector= nn.Linear(concatenated_enc_out.size(-1), path_components.seq_len, bias=True).to(concatenated_enc_out.device)

        # Projection to the dynamically determined sequence length
        dec_out = path_components.dynamic_projector(concatenated_enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, path_components.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, path_components.seq_len, 1))
        return dec_out
        
    def forward(self, x_enc_list, x_mark_enc_list, mask=None):
        encoded_outputs = []
        means_list, stdev_list, N_list = [], [], []

        for x_enc, x_mark_enc, path_components in zip(x_enc_list, x_mark_enc_list, self.paths):
            # x_enc, x_mark_enc = data_preProcessing(x_enc, x_mark_enc)
            enc_out, means, stdev, N = self.encode(x_enc, x_mark_enc, path_components)
            encoded_outputs.append(enc_out)
            means_list.append(means)
            stdev_list.append(stdev)
            N_list.append(N)

        # if encoded_outputs[0].size(-1) != encoded_outputs[1].size(-1):
        #     if encoded_outputs[0].size(-1) < encoded_outputs[1].size(-1):
        #         encoded_outputs[0] = F.pad(encoded_outputs[0], (0, encoded_outputs[1].size(-1) - encoded_outputs[0].size(-1)), "constant", 0)
        # # print(encoded_outputs[0].size(), encoded_outputs[1].size())
        # concatenated_enc_out = torch.cat(encoded_outputs, dim=1)
        
        # Find the maximum sequence length
        max_seq_len = max([enc_out.size(-1) for enc_out in encoded_outputs])
        # print the length for each encoded_outputs
        # Pad the shorter sequences to match the maximum sequence length
        for i in range(len(encoded_outputs)):
            enc_out = encoded_outputs[i]
            seq_len = enc_out.size(-1)
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                encoded_outputs[i] = F.pad(enc_out, (0, pad_len), "constant", 0)
        # print the shape of the encoded_outputs
        concatenated_enc_out = torch.cat(encoded_outputs, dim=1)
        # print the shape of the concatenated_enc_out
        res = []
        for lp in range(len(means_list)):
            means = means_list[lp]
            stdev = stdev_list[lp]
            N = N_list[lp]
            path_components = self.paths[lp]
            dec_out = self.decode(concatenated_enc_out, means, stdev, N, path_components)
            res.append(dec_out)
        return res

    def freeze_path(self, path_index):
        """
        Freeze the parameters of the path at the specified index.
        """
        path = self.paths[path_index]
        for param in path.parameters():
            param.requires_grad = False

    def unfreeze_path(self, path_index):
        """
        Unfreeze the parameters of the path at the specified index.
        """
        path = self.paths[path_index]
        for param in path.parameters():
            param.requires_grad = True


class PathComponents(nn.Module):
    """ Custom module to hold components for each path """
    def __init__(self, configs,d_model):
        self.d_model = d_model
        self.seq_len = configs.seq_len
        super(PathComponents, self).__init__()
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # self.dynamic_projector= nn.Linear(self.d_model, configs.seq_len, bias=True)
        # Usage in your model setup
        self.dynamic_projector = EnhancedDynamicProjectorWithAttention(input_dim=self.d_model, output_dim=configs.seq_len, num_layers=2, bias=True)


class EnhancedDynamicProjectorWithAttention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=2, num_heads=8, bias=True, dropout=0.1):
        super(EnhancedDynamicProjectorWithAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 4

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):  # One less because of the output projection layer
            self.layers.append(nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.layers.append(nn.ReLU())

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> batch,num_variate, d_model
        # Assuming x is of shape (batch_size, sequence_length, input_dim)
        # Convert x to shape (sequence_length, batch_size, input_dim) for nn.MultiheadAttention
        x = x.permute(1, 0, 2)
        x = self.input_projection(x)
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                x, _ = layer(x, x, x)
            else:
                x = layer(x)
            x = self.dropout(x)
        # Convert x back to (batch_size, sequence_length, hidden_dim)
        x = x.permute(1, 0, 2)
        x = self.output_projection(x)
        return x