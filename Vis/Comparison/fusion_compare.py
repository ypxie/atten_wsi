# -*- coding: utf-8 -*-

import os, sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


mpl.style.use('seaborn')

if __name__ == "__main__":
    fontsize = 12
    methods = ['Pooling', 'Self-Att', 'Global-Att', 'Recurrent-Att']
    color_list = ["darkturquoise", "skyblue", "mediumorchid", "steelblue"]
    accuracies = [0.853, 0.880, 0.880, 0.888]
    index = np.arange(len(methods))
    plt.bar(index, accuracies, width=0.6, color=color_list)
    plt.xlabel('Fusion methods', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.ylim(0.84, 0.90)
    plt.xticks(index, methods, fontsize=fontsize, rotation=15)
    plt.title('Comparison of fusion methods on thyroid dataset', fontsize=16)
    plt.tight_layout()
    # plt.autoscale()
    plt.show()
