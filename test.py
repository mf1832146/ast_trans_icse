import random
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def generate_rand_tree(tree_size):
    par_adj = np.zeros((tree_size, tree_size))
    bro_ajd = np.zeros((tree_size, tree_size))

    prufer = [random.randint(0, tree_size-1) for i in range(tree_size - 2)]

    vertices = tree_size
    vertex_set = [0] * tree_size
    for i in range(vertices - 2):
        vertex_set[prufer[i] - 1] += 1

    for i in range(tree_size - 2):
        for j in range(tree_size):
            if vertex_set[j] == 0:
                vertex_set[j] = -1
                par_adj[i-1][j] = 1
                vertex_set[prufer[i] - 1] -= 1
                break

    j = 0

    # For the last element
    end_node = -1
    for i in range(vertices):
        if vertex_set[i] == 0 and j == 0:
            end_node = i
            j += 1
        elif vertex_set[i] == 0 and j == 1:
            par_adj[i][end_node] = 1

    for i in range(vertices):
        current_node = -1
        for j in range(vertices):
            if par_adj[i][j] == 1:
                if current_node > 0:
                    bro_ajd[current_node][j] = 1
                current_node = j

    return par_adj, bro_ajd


x = []
y = []

fig, ax = plt.subplots()

plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')

if __name__ == '__main__':
    k = 5
    run_times = 100000

    results_x = [[]] * k
    results_y = [[]] * k
    for tree_size in range(50, 300):
        par_num = [0] * k
        bro_num = [0] * k
        for a in tqdm(range(run_times)):
            par_ajd, bro_adj = generate_rand_tree(tree_size)
            tmp_par = par_ajd
            tmp_bro = bro_adj
            for cur_k in range(2, k+1):
                tmp_par = np.matmul(tmp_par, par_ajd)
                tmp_bro = np.matmul(tmp_bro, par_ajd)

                par_num[cur_k-1] += np.sum(tmp_par)
                bro_num[cur_k-1] += np.sum(tmp_bro)
        par_num = [p / run_times for p in par_num]
        bro_num = [b / run_times for b in bro_num]

        for cur_k in range(k):
            results_x[k].append(tree_size)
            results_y[k].append(par_num[k])

        for cur_k in range(1, k):
            ax.plot(results_x[cur_k], results_y[cur_k], labe)

    line1, = ax.plot(x, y, label='Using set_dashes()')
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    # Using plot(..., dashes=...) to set the dashing when creating a line
    line2, = ax.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')

    print(par_num, bro_num)