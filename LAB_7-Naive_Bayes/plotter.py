import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("results.json") as file:
        data = json.load(file)

    x = data["x"]
    y = data["y"]

    plot = plt.plot(x, y)
    plt.title("Skutećzność klasyfikacji w zależności od rozmiaru zbioru testowego.")
    plt.xlabel("Wielkość zbioru testowego")
    plt.ylabel("Skuteczność")
    plt.savefig("accuracy_plot.png")
