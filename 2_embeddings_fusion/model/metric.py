# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras


# def calculate_auroc(predictions, labels):
#     """
#     Calculate auroc.
#     :param predictions: predictions
#     :param labels: labels
#     :return: fpr_list, tpr_list, auroc
#     """
#     print(np.max(labels))
#     print(np.min(labels))
#     if np.max(labels) ==1 and np.min(labels)==0:
#         fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
#         auroc = metrics.roc_auc_score(labels, predictions)
#     else:
#         fpr_list, tpr_list = [], []
#         auroc = np.nan
#     return fpr_list, tpr_list, auroc


# def calculate_aupr(predictions, labels):
#     """
#     Calculate aupr.
#     :param predictions: predictions
#     :param labels: labels
#     :return: precision_list, recall_list, aupr
#     """
#     if np.max(labels) == 1 and np.min(labels) == 0:
#         precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
#         aupr = metrics.average_precision_score(labels, predictions)
#     else:
#         precision_list, recall_list = [], []
#         aupr = np.nan
#     return precision_list, recall_list, aupr


def calculate_f1_score(predictions, labels, threshold=0.5):
    predicted_labels = predictions > threshold
    
    # f1_precision = metrics.precision_score(labels, predicted_labels)
    # f1_recall = metrics.recall_score(labels,predicted_labels)
    # f1_score = metrics.f1_score(labels,predicted_labels,zero_division=0)
    tp = np.sum(np.logical_and(predicted_labels, labels))
    fp = np.sum(np.logical_and(predicted_labels, np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(predicted_labels), labels))
    
    if tp + fp == 0:
        f1_precision = 0  
    else:
        f1_precision = tp / (tp + fp)
        
    if tp + fp == 0:
        f1_recall = 0  
    else:
        f1_recall = tp / (tp + fn)
    
    if (f1_precision + f1_recall) == 0:
        f1_score = 0
    else: 
        f1_score = 2 * (f1_precision * f1_recall) / (f1_precision + f1_recall)
    
    return f1_precision,f1_recall,f1_score

def calculate_fmax_score(predictions, labels, beta, threshold=0.8):
    predicted_labels = predictions > threshold
    
    tp = np.sum(np.logical_and(predicted_labels, labels))
    fp = np.sum(np.logical_and(predicted_labels, np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(predicted_labels), labels))
    
    if tp + fp == 0:
        fm_precision = 0  
    else:
        fm_precision = tp / (tp + fp)
        
    if tp + fp == 0:
        fm_recall = 0  
    else:
        fm_recall = tp / (tp + fn)
    
    
    if (beta**2 * fm_precision + fm_recall)==0:
        fmax_score = 0
    else:
        fmax_score = (1 + beta**2) * (fm_precision * fm_recall) / (beta**2 * fm_precision + fm_recall)
    # fm_precision = metrics.precision_score(labels, predicted_labels)
    # fm_recall = metrics.recall_score(labels,predicted_labels)
    # fmax_score = metrics.fbeta_score(labels,predicted_labels,beta=beta,zero_division=0)
    
    return fm_precision,fm_recall,fmax_score



def calculate_auroc(predictions, labels):
    """
    Calculate auroc.
    :param predictions: predictions
    :param labels: labels
    :return: fpr_list, tpr_list, auroc
    """
    
    # print(predictions)
    # print(labels)
    # labels = labels.astype(np.int32)
    fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
    auroc = metrics.roc_auc_score(labels, predictions)
    return fpr_list, tpr_list, auroc


def calculate_aupr(predictions, labels):
    """
    Calculate aupr.
    :param predictions: predictions
    :param labels: labels
    :return: precision_list, recall_list, aupr
    """
    labels = labels.astype(np.int32)
    precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
    aupr = metrics.average_precision_score(labels, predictions)
    return precision_list, recall_list, aupr