"""
Single label accuracy/F1 metrics
"""
import argparse
import os
from io import StringIO

from scipy.special import softmax
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


def classification_f1_report(input_file, label_col, pred_col):
    usecols = [label_col, pred_col]
    col2pos = {x: i for i, x in enumerate(sorted(usecols))} # pos is index in DataFrame from read_csv
    df = pd.read_csv(input_file, header=None, sep="\t",
                     usecols=usecols, quoting=3, dtype=int)
    gold = df.iloc[:, col2pos[label_col]].values
    pred = df.iloc[:, col2pos[pred_col]].values

    del df
    report = classification_report(gold, pred, output_dict=True)
    return report


def classification_f1_auc_report(input_file, label_col, score_col, score_type="score"):
    """
    :param score_col: format is float separated by ,
    :param score_type: "score"|"prob", "score" is scores before softmax; "prob" is normalized proability
    :return:
    """
    assert(score_type)
    usecols = [label_col, score_col]
    col2pos = {x: i for i, x in enumerate(sorted(usecols))} # pos is index in DataFrame from read_csv
    df = pd.read_csv(input_file, header=None, sep="\t",
                     usecols=usecols, quoting=3, dtype=str)

    labels = df.iloc[:, col2pos[label_col]].map(lambda x:int(x)).values

    tmp_file = StringIO("\n".join(df.iloc[:, col2pos[score_col]].tolist()))

    scores = pd.read_csv(tmp_file, sep=",", dtype=float, header=None).values
    if score_type == "score":
        scores = softmax(scores, axis=1)
    else:
        # probs from GPU do not exactly sum up to 1
        scores = scores / np.sum(scores, axis=1)[:, np.newaxis]
    pred_labels = np.argmax(scores, axis=1)
    report = classification_report(labels, pred_labels, output_dict=True)

    # calculate auc
    num_classes = scores.shape[1]
    for i in range(num_classes):
        one_hot_labels = (labels == i)
        auc_for_one_class = roc_auc_score(one_hot_labels, scores[:, i])
        report[str(i)]["auc"] = auc_for_one_class

    for avg_type in ["macro", "weighted"]:
        # ignore samples avg as it is too slow to calculate
        avg_auc = roc_auc_score(labels, scores, average=avg_type, multi_class="ovr")
        if avg_type + " avg" in report:
            report[avg_type + " avg"]["auc"] = avg_auc
        else:
            report[avg_type + " avg"] = {"auc": avg_auc}

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="tsv file",
                        required=True)
    parser.add_argument("--output_file", type=str, help="output file",
                        required=True)
    parser.add_argument("--label_col", type=int, help="which column is ground truth label (0-based)",
                        required=True)
    parser.add_argument("--pred_col", type=int, help="which column is prediction",
                        required=True)
    parser.add_argument("--pred_type", type=str, help="score|label, type of prediction column",
                        choices=["score", "label", "prob"], default="label")
    args = parser.parse_args()

    if args.pred_type == "label":
        report = classification_f1_report(args.input_file, args.label_col, args.pred_col)
    else:
        report = classification_f1_auc_report(args.input_file, args.label_col, args.pred_col, args.pred_type)

    report_df = pd.DataFrame(report).T
    print(report_df)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    report_df.to_csv(args.output_file, sep="\t")