import json

from model import Model


def episodes_test():
    episodes = list(range(500, 5001, 500))
    lr = 0.1
    df = 1.0
    epsilon = 0.5
    temperature = 1.0

    results = {"x": episodes,
               "greedy": list(),
               "epsilon_greedy": list(),
               "boltzmann": list()}
    for e in episodes:
        for s in ['greedy', 'epsilon_greedy', 'boltzmann']:
            model = Model()
            model.train(episodes=e,
                        learning_rate=lr,
                        discount_factor=df,
                        strategy=s,
                        epsilon=epsilon,
                        temperature=temperature)

            mean_reward = model.evaluate(100)
            results[s].append(mean_reward)

    with open('results/episodes.json', "w") as file:
        json.dump(results, file)


def learning_rate_test():
    episodes = 5000
    learning_rates = [0.05 * i for i in range(1, 21)]
    df = 1.0
    epsilon = 0.5
    temperature = 1.0
    results = {"x": learning_rates,
               "greedy": list(),
               "epsilon_greedy": list(),
               "boltzmann": list()}
    for lr in learning_rates:
        for s in ['greedy', 'epsilon_greedy', 'boltzmann']:
            model = Model()
            model.train(episodes=episodes,
                        learning_rate=lr,
                        discount_factor=df,
                        strategy=s,
                        epsilon=epsilon,
                        temperature=temperature)

            mean_reward = model.evaluate(100)
            results[s].append(mean_reward)

    with open('results/learning_rates.json', "w") as file:
        json.dump(results, file)


def discount_factor_test():
    episodes = 5000
    learning_rate = 0.3
    discount_factors = [0.05 * i for i in range(1, 21)]
    epsilon = 0.5
    temperature = 1.0
    results = {"x": discount_factors,
               "greedy": list(),
               "epsilon_greedy": list(),
               "boltzmann": list()}
    for df in discount_factors:
        for s in ['greedy', 'epsilon_greedy', 'boltzmann']:
            model = Model()
            model.train(episodes=episodes,
                        learning_rate=learning_rate,
                        discount_factor=df,
                        strategy=s,
                        epsilon=epsilon,
                        temperature=temperature)

            mean_reward = model.evaluate(100)
            results[s].append(mean_reward)

    with open('results/discount_factors.json', "w") as file:
        json.dump(results, file)


def epsilon_greedy_test():
    episodes = 5000
    learning_rate = 0.3
    discount_factor = 0.5
    epsilons = [0.05 * i for i in range(1, 21)]
    results = {"x": epsilons,
               "epsilon_greedy": list()}
    for e in epsilons:
        model = Model()
        model.train(episodes=episodes,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    strategy='epsilon_greedy',
                    epsilon=e)

        mean_reward = model.evaluate(100)
        results['epsilon_greedy'].append(mean_reward)

    with open('results/epsilon_greedy.json', "w") as file:
        json.dump(results, file)


def boltzmann_test():
    episodes = 5000
    learning_rate = 0.3
    discount_factor = 0.5
    temperatures = list(range(1, 21))
    results = {"x": temperatures,
               "boltzmann": list()}
    for t in temperatures:
        model = Model()
        model.train(episodes=episodes,
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    strategy='boltzmann',
                    temperature=t)

        mean_reward = model.evaluate(100)
        results['boltzmann'].append(mean_reward)

    with open('results/boltzmann.json', "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    # episodes_test()
    # learning_rate_test()
    # discount_factor_test()
    epsilon_greedy_test()
    # boltzmann_test()