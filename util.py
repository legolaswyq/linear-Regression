import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    cost = 1 / 2 / m * np.sum(np.power((np.dot(X, theta) - y), 2))
    return cost


def gradient_descend(X, y, theta, alpha, num_iter):
    m = len(y)
    J_history = np.zeros([num_iter, 1])

    for iter in range(num_iter):
        error = np.dot(X, theta) - y
        theta = theta - alpha / m * np.sum(error * X, axis=0).reshape(-1, 1)
        J_history[iter] = compute_cost(X, y, theta)

    return [J_history, theta]