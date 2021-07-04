import torch
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater
from tqdm import tqdm

from dataset import BaseASTDataSet

__all__ = ['FastASTDataSet']


class FastASTDataSet(BaseASTDataSet):
    def __init__(self, data_dir, max_ast_size, max_nl_len, max_rel_pos, is_ignore, data_set_name, ast_vocab, nl_vocab):
        print('Data Set Name : < Fast AST Data Set >')
        super(FastASTDataSet, self).__init__(
            data_dir, max_ast_size, max_nl_len, max_rel_pos, is_ignore, data_set_name, ast_vocab, nl_vocab
        )

        self.edges_data = self.convert_ast_to_edges()

    def convert_ast_to_edges(self):
        print('building edges.')

        par_edge_data = self.matrices_data['parent']
        bro_edge_data = self.matrices_data['brother']

        edges_data = []

        def edge2list(edges):
            ast_len = min(len(edges), self.max_ast_size)
            start_node = -1 * torch.ones((self.max_rel_pos + 1, self.max_ast_size), dtype=torch.long)
            for key in edges.keys():
                if key[0] < self.max_ast_size and key[1] < self.max_ast_size:
                    value = edges.get(key)
                    if value > self.max_rel_pos and self.ignore_more_than_k:
                        continue
                    value = min(value, self.max_rel_pos)
                    start_node[value][key[1]] = key[0]

            start_node[0][:ast_len] = torch.arange(ast_len)
            return start_node

        for i in tqdm(range(self.data_set_len)):
            par_edges = par_edge_data[i]
            bro_edges = bro_edge_data[i]
            ast_seq = self.ast_data[i]
            nl = self.nl_data[i]

            par_edge_list = edge2list(par_edges)
            bro_edge_list = edge2list(bro_edges)

            ast_vec = self.convert_ast_to_tensor(ast_seq)
            nl_vec = self.convert_nl_to_tensor(nl)

            data = Data(ast_seq=ast_vec,
                        par_edges=par_edge_list,
                        bro_edges=bro_edge_list,
                        nl=nl_vec[:-1],
                        predict=nl_vec[1:])

            edges_data.append(data)

        return edges_data

    def __getitem__(self, index):
        return self.edges_data[index], self.edges_data[index].predict
