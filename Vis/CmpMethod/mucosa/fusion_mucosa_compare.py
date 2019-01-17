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
    accuracies = [0.759, 0.796, 0.815, 0.833]
    index = np.arange(len(methods))
    plt.bar(index, accuracies, width=0.6, color=color_list)
    plt.xlabel('Fusion methods', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.ylim(0.70, 0.85)
    plt.xticks(index, methods, fontsize=fontsize, rotation=15)
    plt.title('Comparison of fusion methods on mucosa dataset', fontsize=16)
    plt.tight_layout()
    # plt.autoscale()
    plt.show()
