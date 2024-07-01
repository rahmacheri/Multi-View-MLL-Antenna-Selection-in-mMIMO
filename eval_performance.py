
import sys
import numpy as np
from collections import Counter
import tensorflow as tf
from parsers import Parser
from config import Config
get_ipython().system('pip install scikit-learn')
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error, label_ranking_loss, hamming_loss, accuracy_score

class eval_performance:
    def __init__(self, config):
        self.config = config

    def patk(self, predictions, labels):
        pak = np.zeros(3)
        K = np.array([1, 3, 5])
        for i in range(predictions.shape[0]):
            pos = np.argsort(-predictions[i, :])
            y = labels[i, :][pos]  # Sort labels based on sorted predictions
            for j in range(3):
                k = K[j]
                pak[j] += np.sum(y[:k]) / k

        pak = (pak / predictions.shape[0])
        return pak * 100.

    def cm_precision_recall(self, prediction, truth):
        """Evaluate confusion matrix, precision and recall for given set of labels and predictions"""
        confusion_matrix = Counter()
        positives = [1]  # Assuming the positive class is 1

        binary_truth = [x in positives for x in truth]
        binary_prediction = [x in positives for x in prediction]

        for t, p in zip(binary_truth, binary_prediction):
            confusion_matrix[(t, p)] += 1

        tp = confusion_matrix[(True, True)]
        tn = confusion_matrix[(False, False)]
        fp = confusion_matrix[(False, True)]
        fn = confusion_matrix[(True, False)]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        return np.array([tp, tn, fp, fn]), precision, recall

    def bipartition_scores(self, labels, predictions):
        """Computes bipartition metrics for given multilabel predictions and labels"""
        sum_cm = np.zeros(4)
        macro_precision = 0
        macro_recall = 0

        for i in range(labels.shape[1]):
            truth = labels[:, i]
            prediction = predictions[:, i]

            cm, precision, recall = self.cm_precision_recall(prediction, truth)

            sum_cm += cm
            macro_precision += precision
            macro_recall += recall

        macro_precision /= labels.shape[1]
        macro_recall /= labels.shape[1]

        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-6)

        micro_precision = sum_cm[0] / (sum_cm[0] + sum_cm[2] + 1e-6)
        micro_recall = sum_cm[0] / (sum_cm[0] + sum_cm[3] + 1e-6)

        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)

        bipartition = np.asarray([micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1])

        return bipartition

    def evaluate(self, predictions, labels):
        '''
        True Positive  :  Label : 1, Prediction : 1
        False Positive :  Label : 0, Prediction : 1
        False Negative :  Label : 1, Prediction : 0
        True Negative  :  Label : 0, Prediction : 0
        '''
        
        assert predictions.shape == labels.shape, "Shapes: %s, %s" % (predictions.shape, labels.shape,)
        metrics = dict()
        metrics['coverage'] = coverage_error(labels, predictions)
        metrics['average_precision'] = label_ranking_average_precision_score(labels, predictions)
        metrics['ranking_loss'] = label_ranking_loss(labels, predictions)

        # Sort predictions for each sample in descending order
        sorted_indices = np.argsort(-predictions, axis=1)
        
        # Initialize selected labels matrix
        selected_labels = np.zeros_like(predictions)
        
        # Set the top k labels with the highest probabilities to 1
        top_k = self.config.solver.top_k  # Get top_k value from config
        for i in range(len(selected_labels)):
            top_indices = sorted_indices[i, :top_k]
            selected_labels[i, top_indices] = 1

        # Compute evaluation metrics
        metrics['bae'] = 0
        metrics['patk'] = self.patk(selected_labels, labels)
        metrics['hamming_loss'] = hamming_loss(y_pred=selected_labels, y_true=labels)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], metrics['macro_precision'],             metrics['macro_recall'], metrics['macro_f1'] = self.bipartition_scores(labels, selected_labels)

        return metrics

