# FAST_paper
Data and code for paper "FAST: Improving Controllability for Text Generation with Feedback Aware Self-Training"

## Data

We provide our train/dev/test splits for public datasets PENS and MReD.

## Code

Most of the models (BART+CTRL, FAST, IPS, etc) are standard generation models implemented with Huggingface transformers, so we didn't provide their code. We provide our implementations for GeDi (adapted from https://github.com/salesforce/GeDi) and PPLM (adapted from https://github.com/uber-research/PPLM).

We also provide some evaluation scripts.