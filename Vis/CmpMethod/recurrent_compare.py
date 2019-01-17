# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    methods = ['2', '3', '4', '5', '6', '7', '8', '9']
    thyroid_accuracies = [0.884, 0.884, 0.888, 0.888, 0.888, 0.884, 0.884, 0.884]
    mucosa_accuracies = [0.815, 0.815, 0.833, 0.833, 0.815, 0.815, 0.815, 0.815]

    with PdfPages('recurrent_steps.pdf') as pdf:
        plt.subplot(2, 1, 1)
        plt.plot(methods, thyroid_accuracies, "gv-", markersize=10, label="Thyroid")
        plt.ylim(0.86, 0.90)
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.title('Comparison of using different recurrent steps', fontsize=15)

        plt.subplot(2, 1, 2)
        plt.plot(methods, mucosa_accuracies, "bv-", markersize=10, label="Mucosa")
        plt.ylim(0.80, 0.85)
        plt.ylabel('Accuracy')
        plt.legend(loc='best')

        plt.xlabel('Number of Recurrent Steps', fontsize=12)
        # plt.show()
        pdf.savefig()
