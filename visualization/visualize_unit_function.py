import matplotlib.pyplot as plt
import numpy as np

from absl import app

plt.style.use("seaborn-whitegrid")

def exp_centered(w):
    return np.exp(w)

def exp_dive_centered(w):
    return np.exp(w) - np.exp(-w)

def plot_unit(min_x, max_x, unit_func, save_name):

    plt.rc('font', size=13)

    step = 0.001

    x = np.arange(min_x, max_x, step)
    y = [unit_func(w) for w in x]

    plt.plot(x, y, color="tab:blue")

    plt.xlim(min_x, max_x)
    plt.ylim(-20, 20)

    plt.xlabel("w")
    if "exu" in save_name:
        plt.ylabel("exp(w)")
    elif "exp_dive" in save_name:
        plt.ylabel("exp(w)-exp(-w)")

    plt.savefig(save_name)
    plt.show()

def main(argv):

    min_x, max_x = -3, 3

    plot_unit(min_x, max_x, exp_centered, "exu.pdf")
    plot_unit(min_x, max_x, exp_dive_centered, "exp_dive.pdf")

if __name__ == '__main__':
    app.run(main)
