from torch_geometric.data.dataloader import Collater

from dataset import BaseASTDataSet
from torch_geometric.data import Data
from tqdm import tqdm
import torch

from utils import UNK, PAD

__all__ = ['GraphDataSet']


class GraphDataSet(BaseASTDataSet):
    def __init__(self, config, data_set_name):
        print('Data Set Name : < Graph Data Set >')
        super(GraphDataSet, self).__init__(config, data_set_name)
        #self.data_set_len = 20
        self.edges_data = self.convert_ast_to_graph()
        self.collector = Collater([], [])

    def __getitem__(self, index):
        return self.edges_data[index], self.edges_data[index].target

    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def convert_ast_to_graph(self):
        par_edge_data = self.matrices_data['parent']
        bro_edge_data = self.matrices_data['brother']

        max_ast_size = self.max_src_len

        edges_data = list()

        for i in tqdm(range(self.data_set_len), desc='building graph from ast...'):
            ast_seq = self.ast_data[i]
            par_edges = par_edge_data[i]
            bro_edges = bro_edge_data[i]
            nl = self.nl_data[i]

            row = list()
            col = list()
            edge_types = list()

            for key in par_edges.keys():
                start_node_index, end_node_index = key
                if start_node_index < max_ast_size and end_node_index < max_ast_size:
                    value = par_edges.get(key)
                    if value != 1:
                        continue
                    row.append(start_node_index)
                    col.append(end_node_index)
                    edge_types.append(1)

            for key in bro_edges.keys():
                start_node_index, end_node_index = key
                if start_node_index < max_ast_size and end_node_index < max_ast_size:
                    value = bro_edges.get(key)
                    if value != 1:
                        continue
                    row.append(start_node_index)
                    col.append(end_node_index)
                    edge_types.append(2)

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(edge_types, dtype=torch.long)

            ast_seq = ast_seq[:self.max_src_len]
            ast_vec = [self.src_vocab.w2i[x] if x in self.src_vocab.w2i else UNK for x in ast_seq]
            ast_vec = ast_vec + [PAD for i in range(self.max_src_len - len(ast_vec))]
            ast_vec = torch.tensor(ast_vec, dtype=torch.long)

            nl = nl[:self.max_tgt_len - 2]
            nl = ['<s>'] + nl + ['</s>']
            nl_vec = [self.tgt_vocab.w2i[x] if x in self.tgt_vocab.w2i else UNK for x in nl]
            nl_vec = nl_vec + [PAD for i in range(self.max_src_len - len(nl_vec))]
            nl_vec = torch.tensor(nl_vec, dtype=torch.long)

            data = Data(x=ast_vec,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        tgt_seq=nl_vec[:-1],
                        target=nl_vec[1:])

            edges_data.append(data)

        return edges_data











