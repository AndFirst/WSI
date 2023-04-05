import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from src.MLP import MLP, NumpyEncoder
from src.main import create_dataset, f, scale_dataset, rescale_results, plot_function
import pygad as pg


def calculate_weights_len(layout):
    weights_sizes = [x * y for x, y in zip(layout[:-1], layout[1:])]
    return sum(weights_sizes)


def calculate_biases_len(layout):
    biases_sizes = [y for y in layout[1:]]
    return sum(biases_sizes)


def calculate_shapes(layout):
    weights_shapes = []
    biases_shapes = []
    for i in range(1, len(layout)):
        w_shape = layout[i - 1], layout[i]
        b_shape = layout[i], 1
        weights_shapes.append(w_shape)
        biases_shapes.append(b_shape)
    return weights_shapes, biases_shapes


def calculate_weights(shapes, weight_values):
    new_weights = []
    i = 0
    for shape in shapes:
        length, width = shape
        layer_size = length * width
        weight = weight_values[i: i + layer_size]
        i += layer_size
        weight = np.reshape([weight], shape).T
        new_weights.append(weight)
    return new_weights


def calculate_biases(shapes, bias_values):
    new_biases = []
    i = 0
    for shape in shapes:
        length = shape[0]
        bias = bias_values[i:i + length]
        i += length
        bias = np.transpose([bias])
        new_biases.append(bias)
    return new_biases


def train(x_train,
          y_train,
          scaler,
          layout: list[int],
          num_generations: int,
          num_parents_mating: int,
          sol_per_pop: int,
          save_model=False):
    def agent_to_network(agent):
        biases_len = calculate_biases_len(layout)

        biases = agent[:biases_len]
        weights = agent[biases_len:]

        weights_shapes, biases_shapes = calculate_shapes(layout)

        new_weights = calculate_weights(weights_shapes, weights)
        new_biases = calculate_biases(biases_shapes, biases)

        network = MLP(layout)
        network.weights = new_weights
        network.biases = new_biases

        return network

    def calculate_error(model):
        y_predicted = np.array([model.predict(xi)[0] for xi in x_train])
        y_predicted = rescale_results(y_predicted, scaler)
        error = -mean_squared_error(y_train, y_predicted)
        return error

    def fitness(agent, idx):
        model = agent_to_network(agent)
        error = calculate_error(model)
        return error

    def on_generation(obj):
        if obj.generations_completed % 100 == 0:
            print(f"Gen {obj.generations_completed}")
            print(f"MSE: {round(obj.best_solutions_fitness[-1], 2)}")
            print(f"Elapsed time: {round(time.time() - begin_stamp, 2)}s\n")

    num_genes = calculate_biases_len(layout) + calculate_weights_len(layout)
    fitness_func = fitness

    solver = pg.GA(num_generations=num_generations,
                   num_parents_mating=num_parents_mating,
                   fitness_func=fitness_func,
                   sol_per_pop=sol_per_pop,
                   num_genes=num_genes,
                   save_solutions=True,
                   on_generation=on_generation)
    begin_stamp = time.time()
    solver.run()

    best_agent = solver.best_solution()[0]
    model = agent_to_network(best_agent)

    if save_model:
        model_name = "-".join(str(i) for i in layout)
        model.save_to_file(model_name + ".json")

    return model


def main(layout,
         num_generations,
         sol_per_pop,
         num_parents_mating):
    x, y = create_dataset((-10.0, 10.0), 0.1, f)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scale_dataset(y, scaler)

    model = train(x, y, scaler, layout, num_generations, num_parents_mating, sol_per_pop)
    y_predict = np.array([model.predict(i) for i in x])
    y_predict = rescale_results(y_predict, scaler)

    result = {
        "x": x.tolist(),
        "y": y.tolist(),
        "y_predict": y_predict.tolist()
    }
    title = f"layout={'-'.join(str(i) for i in layout)} " \
            f"epochs={num_generations} " \
            f"pop_size={sol_per_pop} " \
            f"elite={num_parents_mating}.json"

    with open(title, "w") as file:
        json.dump(result, file, cls=NumpyEncoder)

    return title


if __name__ == "__main__":
    populations = [10, 50, 100, 1000]
    iterations = [100, 500, 1000]
    hiddens = [5, 10, 50, 100]
    for i in iterations:
        for p in populations:
            for h in hiddens:
                layout = (1, h, 1)
                results = main(layout, i, p, p // 4)
                plot_function(results)
