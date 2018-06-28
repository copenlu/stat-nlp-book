import matplotlib.pyplot as plt
import mpld3
import numpy as np


def add_theta(alpha1, theta1, alpha2, theta2):
    theta1_x, theta1_z = theta1
    theta2_x, theta2_z = theta2
    theta_x = {}
    theta_z = {}
    for key in theta1_x.keys():
        theta_x[key] = alpha1 * theta1_x[key] + alpha2 * theta2_x[key]
    for key in theta1_z.keys():
        theta_z[key] = alpha1 * theta1_z[key] + alpha2 * theta2_z[key]
    return theta_x, theta_z


def create_1D_plot(x, loss, theta1, theta2):
    np_loss = np.vectorize(lambda alpha1: loss(add_theta(alpha1, theta1, 1.0 - alpha1, theta2)))
    y = np_loss(x)
    plt.plot(x, y)


def plot_1D(loss, theta1, theta2, N=100, eps=0.01, loss2=None, ylim=None):
    x = np.linspace(eps, 1.0 - eps, N)
    fig = plt.figure()
    if ylim is not None:
        plt.ylim(ylim)
    create_1D_plot(x, loss, theta1, theta2)
    if loss2 is not None:
        create_1D_plot(x, loss2, theta1, theta2)
    return mpld3.display(fig)
