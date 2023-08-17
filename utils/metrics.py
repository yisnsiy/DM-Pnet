import pandas as pd
import torch
import numpy as np
from typing import Union
from sklearn import metrics
from utils.animator import Animator


class Metrics:
    """Evaluate performance of model."""

    @staticmethod
    def accuracy(y_prob: Union[torch.Tensor, np.ndarray, list],
                 y_true: Union[torch.Tensor, np.ndarray, list]) -> float:
        """compute the number of correct predictions.

        Args:
            y_prob: n*m matrix that represent n samples and m class, y_prob[i][j] is
              probability of class j in i-th sample.
            y_true: n*1 matrix that contain n real class.

        Returns:
            number of correct prediction
        """

        assert y_prob is not None or y_true is not None
        if not isinstance(y_prob, torch.Tensor):
            y_prob = torch.tensor(y_prob)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_pred = torch.argmax(y_prob, dim=1)
        elif torch.max(y_true) <= 1.:  # two classification situation
            std_th = .5
            y_pred = y_prob > std_th
            # y_pred = torch.where(y_prob > std_th, 1, 0)
        cmp = (y_pred.type(y_true.dtype) == y_true)
        return float(cmp.type(y_true.dtype).sum())

    @staticmethod
    def evaluate_classification_binary(
        y_prob, 
        y_true, 
        max_f1: bool = False,
        saving_dir: str = None, 
        model_name: str = None, 
        is_plot: bool = True):
        """Evaluate classification binary model."""

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        aupr = metrics.average_precision_score(y_true, y_prob)

        if max_f1 == False:
            th = 0.5
            y_pred = y_prob > th
            f1 = metrics.f1_score(y_true, y_pred)
            precision = metrics.precision_score(y_true, y_pred)
            recall = metrics.recall_score(y_true, y_pred)
            accuracy = metrics.accuracy_score(y_true, y_pred)
            score = dict()
            score['accuracy'] = accuracy
            score['precision'] = precision
            score['f1'] = f1
            score['recall'] = recall
            score['aupr'] = aupr
            score['auc'] = auc
            return score
        scores = []
        for th in thresholds:
            y_pred = y_prob > th
            f1 = metrics.f1_score(y_true, y_pred)
            precision = metrics.precision_score(y_true, y_pred)
            recall = metrics.recall_score(y_true, y_pred)
            accuracy = metrics.accuracy_score(y_true, y_pred)
            score = dict()
            score['accuracy'] = accuracy
            score['precision'] = precision
            score['f1'] = f1
            score['recall'] = recall
            score['th'] = th
            scores.append(score)
        ret = pd.DataFrame(scores)
        print(ret)
        best = ret[ret.f1 == max(ret.f1)]
        th = best.th.values[best.shape[0] - 1]

        best_effect = dict()
        best_effect['accuracy'] = best.accuracy.values[0]
        best_effect['precision'] = best.precision.values[0]
        best_effect['f1'] = best.f1.values[0]
        best_effect['recall'] = best.recall.values[0]
        best_effect['aupr'] = aupr
        best_effect['auc'] = auc
        y_pred = y_prob > th
        if is_plot:
            cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
            Animator.get_confusion_matrix(cnf_matrix, saving_dir, model_name)
            Animator.get_metrics(best_effect, saving_dir, model_name)
            Animator.get_auc(y_prob, y_true, saving_dir, model_name)
            Animator.get_auprc(y_prob, y_true, saving_dir, model_name)
        # print(f'best thresholds is {th}, auc is {auc}, auprc is {aupr}.')
        # print(best)

        return best_effect

