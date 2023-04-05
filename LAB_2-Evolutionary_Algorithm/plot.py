import json
import matplotlib.pyplot as plt


def get_data(path):
    with open(path) as file:
        data = json.load(file)
    return data


def prepare_statistic_data(data):
    x_list = list(data.keys())
    averages = [data[i]["average"] for i in data]
    best_results = [data[i]["best_result"] for i in data]
    return x_list, best_results, averages


def create_convergence_plot(data):
    x_list = list(range(0, len(data["0"])))
    plt.cla()
    for path in data.values():
        plt.plot(x_list, path)


def create_plot(x_list, y_list, y_top, y_bottom, title, xlabel, ylabel):
    plt.cla()
    plt.plot(x_list, y_list)
    plt.ylim(top=y_top, bottom=y_bottom)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"plots/{title}_{xlabel}_{ylabel}.png")


def main():
    f1_c = get_data("results/f1_convergence.json")
    create_convergence_plot(f1_c)
    plt.yscale("log")
    plt.ylim(top=10 ** 12, bottom=10 ** 2)
    plt.savefig("plots/f1_convergence.png")

    f1_gc = get_data("results/f1_gradient_convergence.json")
    create_convergence_plot(f1_gc)
    plt.yscale("log")
    plt.ylim(top=10 ** 5, bottom=4 * 10 ** 2)
    plt.savefig("plots/f1_gradient_convergence.png")

    f9_gc = get_data("results/f9_gradient_convergence.json")
    create_convergence_plot(f9_gc)
    plt.yscale("log")
    plt.ylim(top=10 ** 4, bottom=9 * 10 ** 2)
    plt.savefig("plots/f9_gradient_convergence.png")

    f9_c = get_data("results/f9_convergence.json")
    create_convergence_plot(f9_c)
    plt.yscale("log")
    plt.ylim(top=10 ** 4, bottom=9 * 10 ** 2)
    plt.savefig("plots/f9_convergence.png")

    f1_data = get_data("results/f1_elite_sizes.json")
    x_list, best_results, averages = prepare_statistic_data(f1_data)
    create_plot(x_list, best_results, 130, 100, "f1", "elite size", "best result")
    create_plot(x_list, averages, 4000, 100, "f1", "elite size", "average result")

    f9_data = get_data("results/f9_elite_sizes.json")
    x_list, best_results, averages = prepare_statistic_data(f9_data)
    create_plot(x_list, best_results, 1000, 900, "f9", "elite size", "best result")
    create_plot(x_list, averages, 1300, 900, "f9", "elite size", "average result")

    f1_data = get_data("results/f1_mutation_values.json")
    x_list, best_results, averages = prepare_statistic_data(f1_data)
    create_plot(x_list, best_results, 110, 100, "f1", "mutation_value", "best result")
    create_plot(x_list, averages, 4000, 100, "f1", "mutation_value", "average result")

    f9_data = get_data("results/f9_mutation_values.json")
    x_list, best_results, averages = prepare_statistic_data(f9_data)
    create_plot(x_list, best_results, 910, 900, "f9", "mutation_value", "best result")
    create_plot(x_list, averages, 950, 900, "f9", "mutation_value", "average result")

    f1_data = get_data("results/f1_populations.json")
    x_list, best_results, averages = prepare_statistic_data(f1_data)
    create_plot(x_list, best_results, 200, 100, "f1", "population", "best result")
    create_plot(x_list, averages, 4000, 100, "f1", "population", "average result")

    f9_data = get_data("results/f9_populations.json")
    x_list, best_results, averages = prepare_statistic_data(f9_data)
    create_plot(x_list, best_results, 910, 900, "f9", "population", "best result")
    create_plot(x_list, averages, 950, 900, "f9", "population", "average result")

    f1_data = get_data("results/f1_iterations.json")
    x_list, best_results, averages = prepare_statistic_data(f1_data)
    create_plot(x_list, best_results, 200, 100, "f1", "iterations", "best result")
    create_plot(x_list, averages, 4000, 100, "f1", "iterations", "average result")

    f9_data = get_data("results/f9_iterations.json")
    x_list, best_results, averages = prepare_statistic_data(f9_data)
    create_plot(x_list, best_results, 910, 900, "f9", "iterations", "best result")
    create_plot(x_list, averages, 950, 900, "f9", "iterations", "average result")


if __name__ == "__main__":
    main()
