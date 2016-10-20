import matplotlib.pyplot as plt
import numpy as np
import mpld3

eps = 0.0001


def plot_mle_graph(function,
                   mle_params,
                   x_start=eps, x_end=1 - eps,
                   y_start=eps, y_end=1 - eps, resolution=100,
                   x_label="x", y_label="y",
                   show_constraint=False,
                   show_optimum=False):
    x = np.linspace(x_start, x_end, resolution)
    y = np.linspace(y_start, y_end, resolution)
    xx, yy = np.meshgrid(x, y)
    np_func = np.vectorize(lambda x, y: function(x, y))
    z = np_func(xx, yy)

    optimal_loss = function(*mle_params)
    levels_before = np.arange(optimal_loss - 3.0, optimal_loss, 0.25)
    levels_after = np.arange(optimal_loss, min(optimal_loss + 2.0, -0.1), 0.25)

    fig = plt.figure()
    contour = plt.contour(x, y, z, levels=np.concatenate([levels_before, levels_after]))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if show_constraint:
        plt.plot(x, 1 - x)
    if show_optimum:
        plt.plot(mle_params[0], mle_params[1], 'ro')
    plt.clabel(contour)
    return mpld3.display(fig)


def plot_2D_function(function, x_start=eps, x_end=1 - eps,
                     y_start=eps, y_end=1 - eps, resolution=100,
                     x_label="x", y_label="y"):
    x = np.linspace(x_start, x_end, resolution)
    y = np.linspace(y_start, y_end, resolution)
    xx, yy = np.meshgrid(x, y)
    np_func = np.vectorize(lambda x, y: function(x, y))
    z = np_func(xx, yy)
    fig = plt.figure()
    contour = plt.contour(x, y, z)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.clabel(contour)
    return mpld3.display(fig)
