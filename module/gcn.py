import torch.nn as nn
from torch.nn.modules.transformer import _get_activation_fn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from module import _get_clones


__all__ = ['GCNEncoderLayer', 'GATEncoderLayer', 'TransformerConvEncoderLayer', 'BaseGNNEncoder']


class BaseGNNEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layer, hidden_size):
        super(BaseGNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.layers = _get_clones(encoder_layer, num_layer)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, data):
        x = data.src_emb
        edge_index = data.edge_index
        for layer in self.layers:
            output = layer(x, edge_index)
        output = self.norm(output)
        return output.view(data.batch_size, -1, self.hidden_size)


class BaseGNNEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dim_feed_forward, dropout_rate, activation):
        super(BaseGNNEncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = _get_activation_fn(activation)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.activation(self.dropout(self.conv(x, edge_index)))
        return x


# class ResGCNEncoder(BaseGNNEncoder):
#     def __init__(self, num_layers, hidden_size, dim_feed_forward, dropout_rate, activation):
#         super(ResGCNEncoder, self).__init__()
#         self.enc_layers = _get_clones(ResGCNEncoderLayer(hidden_size, dim_feed_forward, dropout_rate, activation),
#                                       num_layers)
#
#     def forward(self, x, edge_index, edge_attr=None):
#         for layer in self.enc_layers:
#             x = layer(x, edge_index)
#         return x
#
#
# class ResGCNEncoderLayer(nn.Module):
#     def __init__(self, hidden_size, dim_feed_forward, dropout_rate, activation):
#         self.conv = GCNConv(hidden_size, dim_feed_forward, cached=False, normalize=True)
#         self.linear = nn.Linear(dim_feed_forward, hidden_size)
#         self.norm1 = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.norm2 = nn.LayerNorm(hidden_size)
#         self.activation = _get_activation_fn(activation)
#
#     def forward(self, x, edge_index):
#         x = self.norm1(x)
#         output = self.activation(self.dropout(self.conv(x, edge_index)))
#         output = self.norm2(output)
#         output = self.activation(self.dropout(self.linear(output)))
#
#         return x + output


class GCNEncoderLayer(BaseGNNEncoderLayer):
    def __init__(self, hidden_size, dim_feed_forward, dropout_rate, activation):
        super(GCNEncoderLayer, self).__init__(hidden_size, dim_feed_forward, dropout_rate, activation)
        self.conv = GCNConv(hidden_size, hidden_size, cached=False, normalize=True)


class GATEncoderLayer(BaseGNNEncoderLayer):
    def __init__(self, hidden_size, dim_feed_forward, dropout_rate, activation):
        super(GATEncoderLayer, self).__init__(hidden_size, dim_feed_forward, dropout_rate, activation)
        num_heads = dim_feed_forward // hidden_size
        self.conv = GATConv(in_channels=hidden_size, out_channels=hidden_size,  heads=num_heads,
                            dropout=dropout_rate, concat=False)


class TransformerConvEncoderLayer(BaseGNNEncoderLayer):
    def __init__(self, hidden_size, dim_feed_forward, dropout_rate, activation):
        super(TransformerConvEncoderLayer, self).__init__(hidden_size, dim_feed_forward, dropout_rate, activation)
        num_heads = dim_feed_forward // hidden_size
        self.conv = TransformerConv(in_channels=hidden_size, out_channels=hidden_size,
                                    heads=num_heads, dropout=dropout_rate, concat=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = _get_activation_fn(activation)



