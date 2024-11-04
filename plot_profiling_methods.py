import matplotlib.pyplot as plt
import numpy as np

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    n = ["random stopping", "tic-toc", "timeit", "cProfile"]

    ax.scatter(x,y)
        
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i]+.05, y[i]-.05))

    ax.set_xlabel("effort")
    ax.set_ylabel("accuracy")

    plt.savefig("profiling_methods.png")
