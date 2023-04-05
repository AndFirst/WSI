import json
import random
import time
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def sigmoid(x):
    ex = np.exp(-x)
    return 1 / (1 + ex)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cost_derivative(result, expected):
    return 2 * (result - expected)


class MLP:
    def __init__(self,
                 layout: list[int] = None,
                 path: str = None
                 ) -> None:
        if path:
            # init neural network from json file
            self.weights = []
            self.biases = []
            self.load_from_file(path)
            return
        elif not layout:
            # default layout
            layout = [1, 10, 1]

        self.num_of_layers: int = len(layout)
        self.weights: list[np.array] = [np.random.uniform(low=-1.0, high=1.0, size=(y, x)) for x, y in
                                        zip(layout[:-1], layout[1:])]
        self.biases: list[np.array] = [np.random.uniform(low=-1.0, high=1.0, size=(y, 1)) for y in layout[1:]]

    def predict(self, input: float) -> float:
        for bias, weight in zip(self.biases, self.weights):
            input = sigmoid(np.dot(weight, input) + bias)
        return input

    def train(self,
              training_data: list[tuple[float, float]],
              epochs: int,
              batch_size: int,
              learn_rate: float,
              save_model: bool = False
              ) -> None:
        """
        :param training_data: list of pairs (x, y) to be model train set
        :param epochs: number of iterations
        :param batch_size: size of mini batch
        :param learn_rate: hyperparameter of network, says how big will be gradient of change
        :param save_model: if model will be saved after training
        """
        training_data_len = len(training_data)
        begin_stamp = time.time()
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[batch_begin:batch_begin + batch_size]
                            for batch_begin in range(0, training_data_len, batch_size)]
            for mini_batch in mini_batches:
                self._update_network(mini_batch, learn_rate)
            print(f"Epoch: {i}")
            print(f"Elapsed time: {round(time.time() - begin_stamp, 2)}s\n")
        if save_model:
            self.save_to_file("")

    def _update_network(self,
                        mini_batch: list[tuple[float, float]],
                        learn_rate: float
                        ) -> None:
        gradient_b = [np.zeros(bias.shape) for bias in self.biases]
        gradient_w = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self._backpropagation(x, y)
            gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

        self.weights = [w - (learn_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b - (learn_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, gradient_b)]

    def _backpropagation(self,
                         x: float,
                         y: float
                         ) -> tuple[np.array, np.array]:
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        neuron_activation = x
        neuron_activations = [x]
        neurons_results = []

        for bias, weight in zip(self.biases, self.weights):
            neuron_result = np.dot(weight, neuron_activation) + bias
            neurons_results.append(neuron_result)

            neuron_activation = sigmoid(neuron_result)
            neuron_activations.append(neuron_activation)

        delta = cost_derivative(neuron_activations[-1], y) * sigmoid_derivative(neurons_results[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, neuron_activations[-2].T)

        for layer in range(2, self.num_of_layers):
            neuron_result = neurons_results[-layer]
            sp = sigmoid_derivative(neuron_result)
            delta = np.dot(self.weights[1 - layer].T, delta) * sp
            gradient_b[-layer] = delta
            gradient_w[-layer] = np.dot(delta, neuron_activations[-1 - layer].T)
        return gradient_b, gradient_w

    def save_to_file(self, path: str) -> None:
        network_data = {
            "weights": self.weights,
            "biases": self.biases,
        }

        with open(path, "w") as file:
            json.dump(network_data, file, cls=NumpyEncoder)

    def load_from_file(self, path: str) -> None:
        with open(path) as file:
            data = json.load(file)
        weights = data["weights"]
        biases = data["biases"]

        for w, b in zip(weights, biases):
            self.weights.append(np.array(w))
            self.biases.append(np.array(b))
        self.num_of_layers = len(self.biases) + 1
