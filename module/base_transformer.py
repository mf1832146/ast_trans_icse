import torch.nn as nn

from module import Embeddings, FastASTEncoderLayer, FastMultiHeadedAttention, FeedForward, _get_clones, \
    SublayerConnection


class BaseTrans(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, num_heads,
                 max_rel_pos, pos_type, num_layers, dim_feed_forward, dropout, state_dict=None):
        super(BaseTrans, self).__init__()
        self.num_heads = num_heads

        self.pos_type = pos_type.split('_')

        self.src_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=src_vocab_size,
                                        dropout=dropout,
                                        with_pos=False)

        self.tgt_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=tgt_vocab_size,
                                        dropout=dropout,
                                        with_pos=True)


class RobertaEncoder(nn.Module):
    def __init__(self):
        super(RobertaEncoder, self).__init__()


class RobertaEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dim_feed_forward, dropout):
        super(RobertaEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.self_attn = FastMultiHeadedAttention(num_heads, hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, dim_feed_forward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.sublayer = _get_clones(SublayerConnection(hidden_size, dropout), 2)