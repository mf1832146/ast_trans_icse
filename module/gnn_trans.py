import torch.nn as nn

from dataset import make_std_mask
from module import GATEncoderLayer, TransformerConvEncoderLayer, GCNEncoderLayer, BaseGNNEncoder
from module import BaseDecoder, DecoderLayer, Embeddings, Generator
from module import BaseTrans
from utils import PAD


class GNNTrans(BaseTrans):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, num_heads,
                 gnn_type, num_layers, dim_feed_forward, dropout, state_dict=None):
        super(GNNTrans, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.src_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=src_vocab_size,
                                        dropout=dropout,
                                        with_pos=False)

        self.tgt_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=tgt_vocab_size,
                                        dropout=dropout,
                                        with_pos=True)
        if gnn_type == 'gcn':
            encoder_layer = GCNEncoderLayer(hidden_size, dim_feed_forward, dropout, activation="gelu")
        elif gnn_type == 'gat':
            encoder_layer = GATEncoderLayer(hidden_size, dim_feed_forward, dropout, activation="gelu")
        elif gnn_type == 'gnn_trans':
            encoder_layer = TransformerConvEncoderLayer(hidden_size, dim_feed_forward, dropout, activation="gelu")
        else:
            raise Exception('Invalid GNN Type.')
        self.encoder = BaseGNNEncoder(encoder_layer, num_layers, hidden_size)

        decoder_layer = DecoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout, activation="gelu")
        self.decoder = BaseDecoder(decoder_layer, num_layers, norm=nn.LayerNorm(hidden_size))
        self.generator = Generator(tgt_vocab_size, hidden_size, dropout)

        print('Init or load model.')
        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)

    def process_data(self, data):
        data.batch_size = data.ptr.size(0) - 1
        src_seq = data.x  # [batch_size * ast_len]
        data.src_mask = src_seq.view(data.batch_size, -1).eq(PAD)
        data.src_emb = self.src_embedding(src_seq)

        if data.tgt_seq is not None:
            tgt_seq = data.tgt_seq  # [batch_size * nl_len]
            tgt_seq = tgt_seq.view(data.batch_size, -1)  # [batch_size, nl_len]
            data.tgt_mask = make_std_mask(tgt_seq, PAD)
            data.tgt_emb = self.tgt_embedding(tgt_seq)