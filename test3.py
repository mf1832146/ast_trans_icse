import matplotlib.pyplot as plt
import numpy as np


def f(n):
    return 10 * n - 2^(10+2)


if __name__ == '__main__':
    x = range(100, 150)
    y = [f(z) for z in x]
    fig, ax0 = plt.subplots()
    ax0.plot(x, y)
    plt.show()