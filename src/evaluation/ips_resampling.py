import argparse
import os
from collections import Counter
import numpy as np
import pandas as pd
import scipy.special
from io import StringIO


def read_and_get_propensity(input_file, label_col, score_col, score_type="logit"):
    """
    :param score_type: "logit"|"softmax"|"prob", which function for converting score into probability
    :return: propensity: numpy array of float
    """
    usecols = [label_col, score_col]
    col2pos = {x: i for i, x in enumerate(sorted(usecols))}  # pos is index in DataFrame from read_csv
    df = pd.read_csv(input_file, header=None, sep="\t",
                     usecols=usecols, quoting=3, dtype=str)

    tmp_file = StringIO("\n".join(df.iloc[:, col2pos[label_col]].tolist()))
    labels = pd.read_csv(tmp_file, sep=",", dtype=int, header=None).values

    tmp_file = StringIO("\n".join(df.iloc[:, col2pos[score_col]].tolist()))
    scores = pd.read_csv(tmp_file, sep=",", dtype=float, header=None).values

    del df

    if score_type == "logit":
        probs = 1 / (1 + np.exp(-scores))
        log_prob = np.log(probs)
        log_prob_neg = np.log(1-probs)
        del probs
        propensity = np.exp(np.sum(labels * log_prob, axis=1) + np.sum((1-labels) * log_prob_neg, axis=1))
    elif score_type == "softmax":
        probs = scipy.special.softmax(scores, axis=-1)
        propensity = np.choose(labels.flatten(), probs.T)
    elif score_type == "prob":
        # probs from GPU do not exactly sum up to 1
        probs = scores / np.sum(scores, axis=1)[:, np.newaxis]
        propensity = np.choose(labels.flatten(), probs.T)
    else:
        raise ValueError("Unrecognized score_type: %s" % score_type)

    return propensity


def resample_times(propensity, inflation_factor=2, repetition_cap=30, noise_level=0.0, nums_to_sample=None):
    """
    Number of times should be sampled. Support multi-label

    :param propensity: shape (num_rows, )
    :param inflation_factor:
    :param repetition_cap:
    :param noise_level: noise level of the gold label (percentage of wrong labels)
    :param nums_to_sample: None (default)|int; if not None, override inflation_factor

    :return:
    """
    num_rows = len(propensity)

    assert(noise_level < 0.5)
    if noise_level > 0:
        propensity = noise_level + (1 - 2*noise_level) * propensity

    inv_propensity = 1 / propensity
    inv_propensity[np.isnan(inv_propensity)] = 1.0

    if nums_to_sample is not None:
        total_nums_to_sample = nums_to_sample
    else:
        total_nums_to_sample = int(inflation_factor * num_rows)
    batch_size = max(10, total_nums_to_sample // 100)

    num_samples = 0
    repetition_times = np.zeros((num_rows,)) # for each sample, how many times to duplicate it
    while num_samples < total_nums_to_sample:
        inv_propensity = inv_propensity / np.sum(inv_propensity)
        selected_idx = np.random.choice(num_rows, batch_size, p=inv_propensity)
        num_times_per_idx = Counter(selected_idx)
        for k, v in num_times_per_idx.items():
            repetition_times[k] += v

        idx_exceed_cap = repetition_times > repetition_cap
        inv_propensity[idx_exceed_cap] = 0
        repetition_times[idx_exceed_cap] = repetition_cap
        num_samples = repetition_times.sum()

    return repetition_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="tsv file",
                        required=True)
    parser.add_argument("--output_file", type=str, help="output file",
                        required=True)
    parser.add_argument("--output_unsampled_file", type=str, help="if not None, output lines that are not sampled",
                        default=None)
    parser.add_argument("--label_col", type=int, help="which column is ground truth label (0-based)",
                        default=1)
    parser.add_argument("--score_col", type=int, help="which column is prediction score",
                        default=2)
    parser.add_argument("--score_type", type=str,
                        help="logit: binary or multi-label classification; softmax: single-label classification",
                        choices=["logit", "softmax", "prob"], required=True)
    parser.add_argument("--inflation_factor", type=float, help="size of output data / input data",
                        default=2)
    parser.add_argument("--repetition_cap", type=int, help="max times a data point can be resampled",
                        default=30)
    parser.add_argument("--noise_level", type=float, help="noise level for the gold label (default 0)",
                        default=0)
    parser.add_argument("--nums_to_sample", type=int,
                        help="number of data points to sample; will override inflation_factor",
                        default=None)
    parser.add_argument("--seed", type=int, help="random seed",
                        default=None)
    parser.add_argument("--dedup", action="store_true", help="perform a dedup after resampling")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print("Setting np.random.seed to %d" % args.seed)

    propensity = read_and_get_propensity(args.input_file, args.label_col, args.score_col, args.score_type)
    input_data_size = len(propensity)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    rep_times = resample_times(propensity, args.inflation_factor, args.repetition_cap, args.noise_level,
                               args.nums_to_sample)
    if args.dedup:
        print("Performing dedup")
        rep_times[rep_times > 0] = 1

    resampled_data_size = rep_times.sum()
    avg_repetition = rep_times[rep_times > 0].mean()
    dropped_data = np.sum(rep_times == 0)
    print("input_data_size: %d" % input_data_size)
    print("resampled_data_size: %d" % resampled_data_size)
    print("avg_repetition: %f" % avg_repetition)
    print("dropped_data: %d" % dropped_data)
    print("dropped_data percentage: %f" % (dropped_data / input_data_size))

    with open(args.input_file, "r", encoding="UTF-8") as f_in:
        with open(args.output_file, "w", encoding="UTF-8") as f_out:
            for i, l in enumerate(f_in):
                for _ in range(int(rep_times[i])):
                    f_out.write(l.rstrip('\n') + '\n')

        if i != input_data_size - 1:
            raise ValueError("data size mismatch between resampling and writing")

    if args.output_unsampled_file is not None:
        os.makedirs(os.path.dirname(args.output_unsampled_file), exist_ok=True)
        unsampled_lines = 0
        with open(args.input_file, "r", encoding="UTF-8") as f_in:
            with open(args.output_unsampled_file, "w", encoding="UTF-8") as f_out:
                for i, l in enumerate(f_in):
                    if int(rep_times[i]) == 0:
                        f_out.write(l.rstrip('\n') + '\n')
                        unsampled_lines += 1
        print("unsampled_lines: %d" % unsampled_lines)
