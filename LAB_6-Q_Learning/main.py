import json

from matplotlib import pyplot as plt


def plot_comparison(path: str, out_file):
    with open(path) as file:
        result = json.load(file)

    x = result['x']
    greedy = result.get('greedy')
    epsilon_greedy = result.get('epsilon_greedy')
    boltzmann = result.get('boltzmann')

    legend = []
    if greedy:
        plt.plot(x, greedy, 'r')
        legend.append('greedy')
    if epsilon_greedy:
        plt.plot(x, epsilon_greedy, 'g')
        legend.append('epsilon_greedy')
    if boltzmann:
        plt.plot(x, boltzmann, 'b')
        legend.append('boltzmann')

    plt.legend(legend)
    plt.grid(True)
    plt.savefig(out_file)
    plt.close()


if __name__ == "__main__":
    # plot_comparison('results/episodes.json', 'plots/episodes_comparison.png')
    # plot_comparison('results/learning_rates.json', 'plots/lr_comparison.png')
    # plot_comparison('results/discount_factors.json', 'plots/df_comparison.png')
    plot_comparison('results/epsilon_greedy.json', 'plots/epsilon_greedy.png')
    # plot_comparison('results/boltzmann.json', 'plots/boltzmann.png')
