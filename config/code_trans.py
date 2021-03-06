from pathlib import Path

from dataset import BaseCodeDataSet
from module.code_trans import CodeTrans
from utils import LabelSmoothing, PAD

use_clearml = True
project_name = 'baselines'
task_name = 'code_trans'
is_split = True  # only action when data_type in ['sbt', 'pot']
is_test = False

seed = 2021
# data
data_dir = '../data_set/processed/java'
max_tgt_len = 30
max_src_len = 150
data_type = 'code'

num_heads = 8
pos_type = 'p2q_p2k_p2v'
max_rel_pos = 5
num_layers = 4
hidden_size = 256
dim_feed_forward = 2048
is_ignore = True
dropout = 0.2

# train
batch_size = 32
num_epochs = 500
num_threads = 2
config_filepath = Path('./config/code_trans.py')
es_patience = 5
load_epoch_path = ''
val_interval = 5
data_set = BaseCodeDataSet
model = CodeTrans
fast_mod = False
logger = ['tensorboard', 'clear_ml']

# optimizer
learning_rate = 1e-3
warmup = 0.01

# criterion
criterion = LabelSmoothing(padding_idx=PAD, smoothing=0.1)
checkpoint = None
