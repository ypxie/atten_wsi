# -*- coding: utf-8 -*-

import os, sys
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

json_path = "../data/Thyroid/thyroid_pred_gt.json"
categories = ["Benign", "Uncertain", "Malignant"]
n_classes = len(categories)

with open(json_path) as fp:
    data_dict = json.load(fp)


y_gts = data_dict["gts"]
y_gts = np.argmax(np.asarray(y_gts), axis=1)
y_feas = np.asarray(data_dict["feas"])


tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
y_2d = tsne.fit_transform(y_feas)

with PdfPages('thyroid_roc.pdf') as pdf:
    for ind, category in enumerate(categories):
        indices = [i for i, x in enumerate(y_gts) if x == ind]
        plt.scatter(y_2d[indices, 0], y_2d[indices, 1], label=category)
    plt.legend()
    # plt.show()
    pdf.savefig()
