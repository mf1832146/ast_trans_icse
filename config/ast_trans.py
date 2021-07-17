from datetime import datetime
from pathlib import Path
import torch
from dataset.fast_ast_data_set import FastASTDataSet
from module import FastASTTrans
import os
from utils import LabelSmoothing, PAD

use_clearml = True
project_name = 'ast_trans_fast'
task_name = 'ast_trans_fast'

seed = 2021
# data
data_dir = '../data_set/processed/java'
max_tgt_len = 30
max_src_len = 200
data_type = 'ast'

is_split = True

# model
hype_parameters = {
    'pos_type': 'p2q_p2k_p2v',  # ['', 'p2q_p2k', 'p2q_p2k_p2v']
    'par_heads': 4,   # [0,8]
    'max_rel_pos': 5,  # [1, 3, 5, 7]
    'num_layers': 4,  # [2, 4, 6]
    'data_dir': '../data_set/processed/py',  # java, py
    'is_split': True  # need split
}


num_heads = 8
pos_type = ''
par_heads = 4
max_rel_pos = 3
num_layers = 2
hidden_size = 256
dim_feed_forward = 2048
is_ignore = True
dropout = 0.2

# train
batch_size = 32
num_epochs = 500
num_threads = 0
#output_path = Path('./checkpoint/ast_trans/' + datetime.now().strftime("%Y%m%d-%H%M%S"))
output_path = Path('/home/tangze/ast_trans_6_30/clearml/clearml/20210717-001432')
config_filepath = Path('./config/ast_trans.py')
es_patience = 20
load_epoch_path = ''
val_interval = 5
data_set = FastASTDataSet
model = FastASTTrans
fast_mod = False
logger = ['tensorboard', 'clear_ml']

# optimizer
optimizer = 'Adam'
learning_rate = 0.056
reg_scale = 3e-5

# criterion
criterion = LabelSmoothing(padding_idx=PAD, smoothing=0.1)

g = '0, 1, 2, 3'

if g != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = g
    device = "cuda"
    if len(g.split(',')) > 1:
        multi_gpu = True
        batch_size = batch_size * len(g.split(','))
    else:
        multi_gpu = False
else:
    device = 'cpu'
    multi_gpu = False


# src_vocab, _, tgt_vocab = load_vocab(data_dir, is_split)
#
checkpoint = None
# if load_epoch_path != '':
#     file_name = load_epoch_path
#     if torch.cuda.is_available():
#         checkpoint = torch.load(file_name)
#     else:
#         checkpoint = torch.load(file_name, map_location='cpu')








