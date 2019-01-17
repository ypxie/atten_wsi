# -*- coding: utf-8 -*-

import os, sys
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


json_path = "../data/Thyroid/thyroid_pred_gt.json"
categories = ["Benign", "Uncertain", "Malignant"]
n_classes = len(categories)

with open(json_path) as fp:
    data_dict = json.load(fp)


y_test = data_dict["gts"]
y_score = data_dict["preds"]

# Compute ROC curve and ROC area for each class
y_test = np.asarray(y_test)
y_score = np.asarray(y_score)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# drawing

plt.figure(figsize=(16, 5))
with PdfPages('thyroid_roc.pdf') as pdf:
    lw = 3

    for ind, category in enumerate(categories):
        plt.subplot(1, n_classes, ind+1)
        plt.plot(fpr[ind], tpr[ind], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[ind])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {}'.format(category))
        plt.legend(loc="lower right")
    # plt.show()
    pdf.savefig()
