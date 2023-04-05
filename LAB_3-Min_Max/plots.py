import json
import matplotlib.pyplot as plt


def load_data(path: str) -> dict[str, dict[str, int]]:
    with open(path) as file:
        data = json.load(file)
    return data


def main():
    data = load_data("results.json")
    for label, small_data in data.items():
        plt.cla()
        x_labels = list(small_data.keys())
        values = list(small_data.values())
        plt.bar(x_labels, values)
        plt.ylim(bottom=0, top=50)
        plt.title(f"X-depth, Y-depth = {label}")
        plt.savefig(f"plots/{label}")


if __name__ == "__main__":
    main()
