import numpy as np
from numpy import ndarray


def classify(pred_out: ndarray, trsh: float = 0.5):
    pred_round = []
    for val in pred_out:
        if val > trsh:
            pred_round.append(1)
        else:
            pred_round.append(0)
    return np.asarray(pred_round)


def confusion_matrix(output: ndarray, pred_round: ndarray):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for out, pred in zip(output,pred_round):

        if pred == 1:
            if out == 1:
                TP += 1
            else:
                FP += 1
        else:
            if out == 1:
                FN += 1
            else:
                TN += 1
    return {"TN":TN,"TP":TP,"FN":FN,"FP":FP}


def scores(output: ndarray, pred_round: ndarray):
    conf_mat = confusion_matrix(output, pred_round)
    TP = conf_mat["TP"]
    FN = conf_mat["FN"]
    TN = conf_mat["TN"]
    FP = conf_mat["FP"]
    if TP == 0:
        TPR = 0
    else:
        TPR = TP/(TP+FN)

    recall = TPR

    if FP == 0:
        FPR = 0
    else:
        FPR = FP/(FP+TN)

    if TP == 0:
        precision = 0
    else:
        precision = TP/(TP+FP)

    return TPR, FPR, recall, precision
