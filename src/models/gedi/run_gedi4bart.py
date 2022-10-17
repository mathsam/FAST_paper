"""
Run inference with gedi4bart model
"""

import argparse
import os

from gedi4bart import BartWithGedi

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
#from tools.gedi4bart import BartWithGedi

num_beams = 5

tok = BartTokenizer.from_pretrained("facebook/bart-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_gedi_encoder_text(ctrl_and_context, mode="cwd"):
    delimiter_idx = ctrl_and_context.find("§")
    ctrl_code = ctrl_and_context[:delimiter_idx].strip()
    if mode == "cwd":
        context_text = ctrl_and_context[delimiter_idx+1:].strip()
        encoder_text = [c + " § " + context_text for c in ALL_CATEGORIES]
    else:
        encoder_text = ALL_CATEGORIES
    return encoder_text, cat2idx[ctrl_code]

def get_context_text(ctrl_and_context):
    delimiter_idx = ctrl_and_context.find("§")
    return ctrl_and_context[delimiter_idx+1:].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="tsv file",
                        required=True)
    parser.add_argument("--output_file", type=str, help="output file",
                        required=True)
    parser.add_argument("--src_col", type=int, help="which column is source sequence",
                        default=0)
    parser.add_argument("--clm_path", type=str, help="path to conditional language model for generation",
                        required=True)
    parser.add_argument("--gedi_path", type=str, help="path to gedi for steering generation with constrast weight",
                        default=None)
    parser.add_argument("--mode", help="Gedi for contrast-weighted decoding, decide if include control code in src",
                        default="cwd", choices=["cwd", "gedi"])
    parser.add_argument("--omega", type=float, help="parameter to tune up steering",
                        default=30.0)
    parser.add_argument("--input_max_length", type=int, help="max number of input tokens",
                        default=128)
    parser.add_argument("--guide_input_max_length", type=int, help="max number of input tokens for guiding model",
                        default=None)
    parser.add_argument("--output_max_length", type=int, help="max number of output tokens",
                        default=32)
    parser.add_argument("--fp16", action="store_true", help="use float 16 precision")
    parser.add_argument("--ds", choices=["pens", "ads", "mred"], help="which data set", default="pens")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if args.guide_input_max_length is None:
        args.guide_input_max_length = args.input_max_length

    print("ds = %s" %args.ds)
    if args.ds == "pens":
        ALL_CATEGORIES = ["short", "long"]
    elif args.ds == "ads":
        ALL_CATEGORIES = ['Product or Service', 'Location', 'Inventory and Selection', 'Call to Action',
                          'Advertiser Name or Brand', 'Price and Fees', 'Benefit', 'Customer Problem', 'Highlight']
    else:
        ALL_CATEGORIES = ["abstract", "strength", "weakness", "rating_summary", "ac_disagreement",
                          "rebuttal_process", "suggestion", "decision", "misc"]

    cat2idx = {c: i for i, c in enumerate(ALL_CATEGORIES)}

    if args.fp16:
        print("Using fp16 precision")
    print("mode = %s" %args.mode)

    clm = BartWithGedi.from_pretrained(args.clm_path).to(device)
    if args.fp16:
        clm = clm.half()

    if args.gedi_path is not None:
        gedi = BartForConditionalGeneration.from_pretrained(args.gedi_path).to(device)
        if args.fp16:
            gedi = gedi.half()
    else:
        gedi = None

    cache = dict()
    num_input_rows = 0
    with open(args.input_file, "r", encoding="UTF-8") as f_in:
        with open(args.output_file, "w", encoding="UTF-8") as f_out:
            for l in f_in:
                num_input_rows += 1
                cols = l.split('\t')
                cols[-1] = cols[-1].strip()  # remove line break in the end
                text = cols[args.src_col].strip()
                if len(text) == 0:
                    raise RuntimeError("empty source sequence")
                
                if not text in cache:
                    if args.mode == "cwd":
                        clm_input = tok(text, return_tensors="pt", max_length=args.input_max_length, truncation=True)
                    else:
                        clm_input = tok(get_context_text(text), return_tensors="pt", max_length=args.input_max_length, truncation=True)
                    if gedi is not None:
                        gedi_inputs, desired_ctrl_idx = prepare_gedi_encoder_text(text, mode=args.mode)
                        gedi_batch = tok(gedi_inputs, return_tensors="pt", max_length=args.guide_input_max_length,
                            truncation=True, padding=True)

                        generated_ids = clm.generate(
                            clm_input["input_ids"].to(device),
                            max_length=args.output_max_length,
                            num_beams=num_beams,
                            min_length=5,
                            gedi_model = gedi,
                            gedi_input_ids = gedi_batch["input_ids"].to(device),
                            desired_ctrl_idx = desired_ctrl_idx,
                            disc_weight=args.omega)
                    else:
                        generated_ids = clm.generate(
                            clm_input["input_ids"].to(device),
                            max_length=args.output_max_length,
                            num_beams=num_beams,
                            min_length=5)

                    gen_text = tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    cache[text] = gen_text
                else:
                    gen_text = cache[text]

                f_out.write("\t".join(cols + [gen_text]) + "\n")

                if num_input_rows % 100 == 0:
                    print("Processed rows %d" %num_input_rows)

    print("num_input_rows: %d" % num_input_rows)