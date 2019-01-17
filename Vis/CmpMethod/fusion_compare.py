# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    fontsize = 10

    methods = ['Pooling', 'Self-Att', 'Global-Att', 'Recurrent-Att']
    color_list = ["olivedrab", "skyblue", "mediumorchid", "steelblue"]
    index = np.arange(len(methods))

    thyroid_accuracies = [0.853, 0.880, 0.880, 0.888]
    mucosa_accuracies = [0.759, 0.796, 0.815, 0.833]
    with PdfPages('fusion_methods.pdf') as pdf:
        plt.subplot(2, 1, 1)
        plt.bar(index, thyroid_accuracies, width=0.6, color=color_list, label="Thyroid")
        plt.xticks(index, methods, fontsize=fontsize, rotation=5)
        plt.ylabel('Accuracy', fontsize=fontsize)
        plt.ylim(0.84, 0.90)
        plt.legend(loc='upper left')
        plt.title('Comparison of different fusion methods', fontsize=16)

        plt.subplot(2, 1, 2)
        plt.bar(index, mucosa_accuracies, width=0.6, color=color_list, label="Mucosa")
        plt.xticks(index, methods, fontsize=fontsize, rotation=5)
        plt.ylabel('Accuracy', fontsize=fontsize)
        plt.ylim(0.72, 0.85)
        plt.xlabel('Fusion method')
        plt.legend(loc='upper left')

        # plt.show()
        pdf.savefig()
