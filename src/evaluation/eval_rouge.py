import argparse
import os
import pandas as pd
import argparse
import os
import pandas as pd
from rouge import Rouge as RougeMetrics # use py-rouge package


def cal_rouge(input_file, gold_col, pred_col):
    usecols = [gold_col, pred_col]
    col2pos = {x: i for i, x in enumerate(sorted(usecols))}  # pos is index in DataFrame from read_csv
    df = pd.read_csv(input_file, header=None, sep="\t",
                     usecols=usecols, quoting=3, dtype=str)
    df.fillna("EOS", inplace=True) # in case ground truth or prediction is an empty string
    df = df.applymap(lambda x: x.lower())
    golds = df.iloc[:, col2pos[gold_col]].tolist()
    preds = df.iloc[:, col2pos[pred_col]].tolist()

    rouge = RougeMetrics(metrics=['rouge-n', 'rouge-l'], max_n=2)
    scores = rouge.get_scores(preds, golds)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="tsv file",
                        required=True)
    parser.add_argument("--output_dir", type=str, help="output directory",
                        required=True)
    parser.add_argument("--gold_tgt_col", type=int, help="which column is gold (groundtruth) target sequence",
                        default=0)
    parser.add_argument("--pred_tgt_col", type=int, help="which column is generated target sequence",
                        default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rouge_rpt = cal_rouge(args.input_file, args.gold_tgt_col, args.pred_tgt_col)
    rouge_rpt = pd.DataFrame(rouge_rpt).T
    rouge_rpt = rouge_rpt.sort_index()
    print(rouge_rpt)
    rouge_rpt.to_csv(os.path.join(args.output_dir, "rouge_rpt.tsv"), sep="\t")