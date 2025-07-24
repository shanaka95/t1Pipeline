import numpy as np
from action_recognition.ctrgcn.graph import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 0), (2, 1), (3, 2), (4, 3),      # spine chain
    (5, 1), (6, 5), (7, 6),             # left arm
    (8, 1), (9, 8), (10, 9),            # right arm  
    (11, 0), (12, 11), (13, 12),        # left leg
    (14, 0), (15, 14), (16, 15)         # right leg
]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    # os.environ['DISPLAY'] = 'localhost:10.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i, graph in enumerate(A):
        plt.figure()
        plt.imshow(graph, cmap='gray')
    plt.show()
    print("OK") 