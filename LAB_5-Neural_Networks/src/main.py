import json
from typing import Callable
from numpy import ndarray
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from MLP import MLP, NumpyEncoder
import numpy as np

def f(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


def create_dataset(domain: tuple[float, float], step_size: float, function: Callable) -> tuple[ndarray, ndarray]:
    min_x, max_x = domain
    n_samples = int(abs(max_x - min_x) / step_size) + 1
    x_values = np.linspace(max_x, min_x, n_samples)
    y_values = function(x_values)
    return x_values, y_values


def scale_dataset(y_values: ndarray, scaler):
    y_values = y_values.reshape((len(y_values), 1))
    y_values = scaler.fit_transform(y_values)
    y_values = y_values.flatten()
    return y_values


def rescale_results(y_values: ndarray, scaler):
    y_values = y_values.reshape((len(y_values), 1))
    y_values = scaler.inverse_transform(y_values)
    y_values = y_values.flatten()
    return y_values


def main(epochs: int, hidden_size: int):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x, y = create_dataset((-10.0, 10.0), 0.001, f)
    y_scaled = scale_dataset(y, scaler)

    learning_rate = 2.0
    batch_size = 10

    model = MLP([1, hidden_size, 1])
    # model = MLP(path="1.json")
    train_set = [(a, b) for a, b in zip(x, y_scaled)]
    model.train(train_set, epochs, batch_size, learning_rate)

    y_predict = np.array([model.predict(i) for i in x])
    y_predict = rescale_results(y_predict, scaler)

    result = {
        "x": x.tolist(),
        "y": y.tolist(),
        "y_predict": y_predict.tolist()
    }
    title = f"e={epochs} hidden={hidden_size}.json"
    with open(title, "w") as file:
        json.dump(result, file, cls=NumpyEncoder)

    return title


def plot_function(results_filename: str):
    with open(results_filename) as file:
        result = json.load(file)

    x = result["x"]
    y = result["y"]
    y_predict = result["y_predict"]
    avg_mse = mean_squared_error(y, y_predict)

    title = results_filename.replace(".json", "").replace("./", "")

    plt.plot(x, y, "r")
    plt.plot(x, y_predict, "b")
    plt.legend(["Real value", "Prediction"])
    plt.title(f"{title} MSE={round(avg_mse, 2)}")
    plt.grid(True)

    plt.savefig(f"{title}.png")
    plt.close()


if __name__ == "__main__":
    # epochs = [5, 10, 25, 50, 75, 100, 150, 200]
    # hiddens = [5, 10, 25, 50, 75, 100, 150, 200]

    # for epoch in epochs:
    #    for hidden in hiddens:
    #        title = main(epoch, hidden)
    #        plot_function(title)

    with open("results.json") as file:
        result = json.load(file)

    x = result["best_of_epochs"]["epochs"]
    x = list(map(lambda a: str(a), x))
    y = result["best_of_epochs"]["best"]
    title = "best of epochs"
    plt.xlabel("epochs")
    plt.ylabel("avg mse")
    plt.title(title)
    plt.plot(x, y)
    plt.grid(True)

    plt.savefig(f"{title}.png")
    plt.close()

    x = result["best_of_hidden"]["hidden"]
    x = list(map(lambda a: str(a), x))
    y = result["best_of_hidden"]["best"]
    title = "best of hidden"
    plt.xlabel("hidden size")
    plt.ylabel("avg mse")
    plt.title(title)
    plt.plot(x, y)
    plt.grid(True)

    plt.savefig(f"{title}.png")
    plt.close()
