import time
import matplotlib.pyplot as plt
import numpy as np
import autograd as ad


def solver(f, x0, iterations, step_size):
    """
    :param f: target function
    :param x0: start point (vector) of algorithm
    :param iterations: number of iterations algorithm go on
    :param step_size: size of step to next x point
    :return: final point, route of function values
    """
    grad_func = ad.grad(f)
    values = [f(x0), ]
    for i in range(iterations):
        step = step_size * grad_func(x0)
        x0 -= step
        values.append(f(x0))
    return x0, values


def q(x):
    q_sum = 0.0
    n = len(x)
    for i in range(0, n):
        q_sum += pow(q_param, (i / (n - 1))) * pow(x[i], 2)
    return q_sum


def is_minimum(point, epsilon):
    """
    :param point: vector of floats
    :param epsilon: precision of checking minimum
    :return: if all absolute values of point coordinate is smaller than epsilon
    """
    for number in point:
        # number != number checks if number is not 'infinity'
        if abs(number) > epsilon or number != number:
            return False
    return True


if __name__ == '__main__':
    # params of algorithm test
    vector = np.random.uniform(-100, 100, 10)
    steps_list = [1000, 2000, ]
    step_values = np.array([0.001, 0.01, 0.1])
    q_params = np.array([1.0, 10.0, 100.0, ])

    for steps in steps_list:
        for q_param in q_params:
            for step_value in step_values:
                start = time.time()
                result, function_values = solver(q, vector.copy(), steps, step_value)
                end = time.time()
                execution_time = end - start
                print(f"steps = {steps}\n"
                      f"step_value = {step_value}\n"
                      f"q_param = {q_param}\n"
                      f"execution_time = {execution_time}\n"
                      f"result = {result}\n"
                      f"minimum_found = {is_minimum(result, 1.0)}\n"
                      f"f(last) = {function_values[-1]}\n")

                # plot
                x_axis = list(range(0, steps + 1))
                plt.cla()
                plt.plot(x_axis, function_values)
                plt.ylabel('q(xt)', rotation=0, loc="top")
                plt.xlabel('t', loc="right")
                plt.suptitle(f"α = {q_param}, step_size={step_value}, steps={steps}")
                plt.title(f"found={is_minimum(result, 1.0)}, f(last)={function_values[-1]}", ha="center")
                plt.savefig(f"plots/{steps}_{q_param}_{step_value}.png")
