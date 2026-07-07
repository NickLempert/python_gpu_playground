from typing import Sequence

import numba
import numpy as np
from numba import cuda

from utils import gpu_program


@numba.vectorize(['float32(float32)'], target='cuda')
def erase(a):
    return 0


@numba.vectorize(['float32(float32)'], target='cuda')
def relu(a):
    return a


@numba.vectorize(['float32(float32)'], target='cuda')
def sigmoid(a):
    return 1/(1+np.e**(-a))


class NeuralNetwork:

    def __init__(self, structure: Sequence[int], activation_function=relu):

        self.structure_local = np.array(structure)

        self.structure = cuda.to_device(self.structure_local)
        self.nodes = cuda.to_device(np.zeros(np.sum(self.structure), dtype=np.float32))
        self.weights = cuda.to_device(np.ones(sum((self.structure[i] * self.structure[i-1]
                                                    for i in range(1, len(structure)))), dtype=np.float32))
        self.biases = cuda.to_device(np.zeros(len(self.nodes), dtype=np.float32))

        self.activation_function = activation_function

    @staticmethod
    @gpu_program(['int32, int32[:], int32, int32, float32[:], float32[:], float32[:]'],
                 dimensions=1, default_shape_source=5)
    def progress(layer, structure, first_connection, first_out_node, nodes, weights, biases):
        off = cuda.grid(1)

        if off >= structure[layer]*structure[layer-1]:
            return

        connection = first_connection + off

        node_off = off // structure[layer-1] % structure[layer]

        current_out_node = first_out_node + node_off
        current_in_node = first_out_node - structure[layer] + off % structure[layer-1]

        print(current_in_node, current_out_node, connection)
        for _ in range(off):
            cuda.atomic.add(nodes, current_out_node, nodes[current_in_node] * weights[connection])

        if off % structure[layer-1] == 0:
            cuda.atomic.add(nodes, current_out_node, biases[current_out_node])

    def __call__(self, *args: float):
        erase(self.nodes, out=self.nodes)
        for ind, arg in enumerate(args):
            self.nodes[ind] = arg
        layer = 1
        first_connection = 0
        first_out_node = 0
        while layer < len(self.structure_local):
            first_out_node += self.structure_local[layer-1]

            self.progress(layer,
                          self.structure,
                          first_connection,
                          first_out_node,
                          self.nodes,
                          self.weights,
                          self.biases)

            first_connection += self.structure_local[layer]*self.structure_local[layer-1]

            layer += 1

        nodes = self.nodes.copy_to_host()

        # return nodes[-self.structure_local[-1]:]
        return nodes


if __name__ == '__main__':
    neural_net = NeuralNetwork([2, 3, 3, 1])

    print(neural_net(1, 1))
