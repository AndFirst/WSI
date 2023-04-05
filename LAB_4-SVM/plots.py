import json

import matplotlib.pyplot as plt


def create_comparison_plots():
    with open("comparison.json") as file:
        data = json.load(file)
    x = [float(key) for key in data["white_lin"].keys()]
    y_lin = list(data["white_lin"].values())
    y_rfb = list(data["white_rfb"].values())

    fig, ax = plt.subplots()
    ax.plot(x, y_lin, label="linear")
    ax.plot(x, y_rfb, label="RFB")

    ax.set_xscale("log")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("C")
    ax.set_title("White wine kernels comparison")
    ax.legend()
    plt.grid()
    fig.savefig("plots/comparison_white.png")
    plt.close(fig)

    y_lin = list(data["red_lin"].values())
    y_rfb = list(data["red_rfb"].values())

    fig, ax = plt.subplots()
    ax.plot(x, y_lin, label="linear")
    ax.plot(x, y_rfb, label="RFB")

    ax.set_xscale("log")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("C")
    ax.set_title("Red wine kernels comparison")
    ax.legend()
    plt.grid()
    fig.savefig("plots/comparison_red.png")
    plt.close(fig)


def create_sigma_plots():
    with open("sigma_comparison.json") as file:
        data = json.load(file)
    x = [float(key) for key in data["white_rfb"].keys()]
    y_white = list(data["white_rfb"].values())
    y_red = list(data["red_rfb"].values())

    fig, ax = plt.subplots()
    ax.plot(x, y_white)
    ax.set_xscale("log")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("sigma")
    ax.set_title("White wine sigma")
    plt.grid()
    fig.savefig("plots/sigma_white.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_red)
    ax.set_xscale("log")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("sigma")
    ax.set_title("Red wine sigma")
    plt.grid()
    fig.savefig("plots/sigma_red.png")
    plt.close(fig)


if __name__ == "__main__":
    create_comparison_plots()
    create_sigma_plots()
