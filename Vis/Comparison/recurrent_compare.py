import os, sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# mpl.style.use('seaborn')


if __name__ == "__main__":
    fontsize = 10
    methods = ['2', '3', '4', '5', '6', '7', '8', '9']
    accuracies = [0.884, 0.884, 0.884, 0.888, 0.884, 0.884, 0.884, 0.884]

    plt.plot(methods, accuracies, marker="v", markersize=10)

    plt.xlabel('Number of Recurrent Steps', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.ylim(0.85, 0.90)
    plt.xticks(methods, fontsize=fontsize)
    plt.title('Comparison of recurrent steps on thyroid dataset', fontsize=16)
    plt.tight_layout()
    # plt.autoscale()
    plt.show()
