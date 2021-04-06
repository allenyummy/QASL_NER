# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Evaluation

import logging
import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import run.globals as globals
from torch.nn import CrossEntropyLoss as CE

logger = logging.getLogger(__name__)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    pad_token_label_id = CE().ignore_index  # -100
    true_predictions = [
        [
            globals.label_list[p]
            for (p, l) in zip(prediction, label)
            if l != pad_token_label_id
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [
            globals.label_list[l]
            for (p, l) in zip(prediction, label)
            if l != pad_token_label_id
        ]
        for prediction, label in zip(predictions, labels)
    ]

    results = globals.metric.compute(
        predictions=true_predictions, references=true_labels
    )

    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value

    final_results["overall_precision"] = results["overall_precision"]
    final_results["overall_recall"] = results["overall_recall"]
    final_results["overall_f1"] = results["overall_f1"]
    final_results["overall_accuracy"] = results["overall_accuracy"]
    return final_results