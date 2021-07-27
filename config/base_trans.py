

__all__ = ['get_model']

from module import FastASTTrans, Code2SeqTrans
from module.code_trans import CodeTrans
from module.gnn_trans import GNNTrans


def get_model(config):
    if config.model == FastASTTrans:
        model = config.model(config.src_vocab.size(), config.tgt_vocab.size(),
                             config.hidden_size,
                             config.par_heads, config.num_heads,
                             config.max_rel_pos,
                             config.pos_type,
                             config.num_layers,
                             config.dim_feed_forward,
                             config.dropout,
                             config.checkpoint)
    if config.model == CodeTrans:
        model = config.model(config.src_vocab.size(), config.tgt_vocab.size(),
                             config.hidden_size,
                             config.num_heads,
                             config.max_rel_pos,
                             config.pos_type,
                             config.num_layers,
                             config.dim_feed_forward,
                             config.dropout,
                             config.checkpoint)
    if config.model == Code2SeqTrans:
        model = config.model(config.src_vocab.size(), config.tgt_vocab.size(),
                             config.hidden_size,
                             config.num_heads,
                             config.max_rel_pos,
                             config.pos_type,
                             config.num_layers,
                             config.dim_feed_forward,
                             config.dropout,
                             config.checkpoint)
    if config.model == GNNTrans:
        model = config.model(config.src_vocab.size(), config.tgt_vocab.size(),
                             config.hidden_size,
                             config.num_heads,
                             config.gnn_type,
                             config.num_layers,
                             config.dim_feed_forward,
                             config.dropout,
                             config.checkpoint)

    return model

