import random
import time
from typing import Sequence

import numba
import numpy as np
from numba import cuda

from utils import gpu_program


@numba.vectorize(['float32(float32)'], target='cuda')
def erase(a):
    return 0


@numba.vectorize(['float32(float32)'], target='cuda')
def none(a):
    return a


@numba.vectorize(['float32(float32)'], target='cuda')
def relu(a):
    return a if a > 0 else 0


@numba.vectorize(['float32(float32)'], target='cuda')
def sigmoid(a):
    return 1/(1+np.e**(-a))


@numba.vectorize(['float32(float32)'], target='cuda')
def tanh(a):
    return np.tanh(a)


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

        current_out_node = first_out_node + off // structure[layer-1] % structure[layer]
        current_in_node = first_out_node - structure[layer-1] + off % structure[layer-1]

        cuda.atomic.add(nodes, current_out_node, nodes[current_in_node] * weights[connection])

        if off % structure[layer-1] == 0:
            cuda.atomic.add(nodes, current_out_node, biases[current_out_node])

    @staticmethod
    @gpu_program(['float32[:], uint32, float32'],
                 dimensions=1)
    def mutate_parameters(parameters, seed, amount):
        off = cuda.grid(1)
        if off < parameters.size:
            seed = (seed * 3266489917) + (off ** 2 * 3266489917)
            seed ^= seed >> 15
            seed *= 2246822519
            seed ^= seed >> 13

            quality = 4096
            parameters[off] += ((seed % quality)/quality-0.5)*2*amount

    def __call__(self, *args: float):
        erase(self.nodes, out=self.nodes)
        for ind, arg in enumerate(args):
            self.nodes[ind] = arg
        layer = 1
        first_connection = 0
        first_out_node = 0
        while layer < len(self.structure_local):
            first_out_node += self.structure_local[layer-1]
            self.activation_function(self.nodes[first_out_node-self.structure_local[layer-1]:first_out_node])
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
        return nodes[-self.structure[-1]:]

    def mutate(self, amount: float, weight_amount=1.0, bias_amount=0.5):
        seed = random.randrange(9999999999)
        self.mutate_parameters(self.weights, seed, amount*weight_amount)
        self.mutate_parameters(self.biases, seed, amount*bias_amount)

    def evolve(self, batch_inputs: Sequence[Sequence[float]], batch_outputs: Sequence[Sequence[float]], amount: float):
        results = []
        for i, o in zip(batch_inputs, batch_outputs):
            results.append(abs(np.array(o)-self(*i)))
        avg_error = sum(map(sum, results))/len(batch_inputs)

        old_weights = self.weights.copy_to_host()
        old_biases = self.biases.copy_to_host()

        self.mutate(amount)

        results = []
        for i, o in zip(batch_inputs, batch_outputs):
            results.append(abs(np.array(o)-self(*i)))
        new_avg_error = sum(map(sum, results))/len(batch_inputs)

        if new_avg_error > avg_error:
            self.weights = cuda.to_device(old_weights)
            self.biases = cuda.to_device(old_biases)

        return min(new_avg_error, avg_error)


if __name__ == '__main__':
    # neural_net = NeuralNetwork([2, 100, 1000, 5000, 5000, 1000, 100, 5])
    # # print('calculating...')
    # # print(neural_net(1, 1))
    # for _ in range(5):
    #     t = time.perf_counter()
    #     r = neural_net(random.uniform(-5, 5), random.uniform(-5, 5))
    #     print(time.perf_counter()-t)
    #     print(r)

    test_network = NeuralNetwork([2, 10, 4, 4, 1], activation_function=relu)
    for step in range(5000):
        error = test_network.evolve([[1, 1], [0, 0], [0.5, 0.5]],
                                    [[1, 1], [-1, -1], [0.5, 0.5]],
                                    random.uniform(0, 5))
        if step % 100 == 0:
            print(error, step)
    print(test_network(1, 1))
    print(test_network(0, 0))
    print(test_network(0.5, 0.5))
