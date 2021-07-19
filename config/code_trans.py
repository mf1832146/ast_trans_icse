from pathlib import Path


use_clearml = True
project_name = 'code_trans'
task_name = 'input_with_code'
is_test = False

seed = 2021
# data
data_dir = '../data_set/processed/java'
max_tgt_len = 30
max_src_len = 150
data_type = 'ast'

num_heads = 8
pos_type = 'p2q_p2k_p2v'
max_rel_pos = 5
num_layers = 4
hidden_size = 256
dim_feed_forward = 2048
is_ignore = True
dropout = 0.2

# train
batch_size = 64
num_epochs = 500
num_threads = 2
config_filepath = Path('./config/ast_trans.py')
es_patience = 20
load_epoch_path = ''
val_interval = 5
data_set = FastASTDataSet
model = FastASTTrans
fast_mod = False
logger = ['tensorboard', 'clear_ml']

# optimizer
learning_rate = 1e-3
warmup = 0.01

# criterion
criterion = LabelSmoothing(padding_idx=PAD, smoothing=0.1)

g = '0'

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

checkpoint = None
