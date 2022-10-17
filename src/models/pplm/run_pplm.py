#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from tqdm import trange

#from pplm_classification_head import ClassificationHead
from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    AutoModel, AutoTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer
)
from transformers.file_utils import cached_path
from classification import classification_head
import os 
import time

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

BAG_OF_WORDS_ARCHIVE_MAP = {
    "legal": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    "military": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    "politics": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    "religion": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    "science": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    "space": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    "technology": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)

def perturb_past(
    past,
    model,
    last,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    one_hot_bows_vectors=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
    is_encoder_decoder=True,
    encoder_outputs=None,
    output_so_far=None
):
    # Generate inital perturbed past
    key_value_lens = [key_value_ele.shape[2] for key_value_ele in past[0]]
    if is_encoder_decoder: 
        grad_accumulator = [torch.concat([torch.zeros_like(_p) for _p in p], 2) for p in past]
    else: 
        grad_accumulator = [torch.zeros_like(p) for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    if is_encoder_decoder: 
        _, _, curr_length, _ = past[0][0].shape
    else: 
        _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(past[0].shape[-1:])

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2]) + tuple([curr_length - window_length]) + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
    else:
        window_mask = torch.ones_like(grad_accumulator[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        #print("Iteration ", i + 1)
        curr_perturbation = [p_.requires_grad_(True).to(device=device) for p_ in grad_accumulator]
        # make sure p_.grad is not None
        for p_ in curr_perturbation:
            p_.retain_grad()

        #print(curr_perturbation[0].mean())
        # Compute hidden using perturbed past
        if is_encoder_decoder: 
            grad_accumulator_split = [torch.split(layer_grad_accumulator, key_value_lens, 2) for layer_grad_accumulator in curr_perturbation]
            perturbed_past = [tuple(map(add, past[layer_id], grad_accumulator_split[layer_id])) for layer_id in range(len(grad_accumulator_split))]
        else: 
            perturbed_past = list(map(add, past, curr_perturbation))
        
        if is_encoder_decoder: 
            pass 
        else:     
            _, _, _, curr_length, _ = curr_perturbation[0].shape
        
        if is_encoder_decoder: 
            lm_output = model(
                decoder_input_ids=last, 
                past_key_values=perturbed_past, 
                encoder_outputs=encoder_outputs
            )
            hidden_states_key = "decoder_hidden_states"
        else: 
            lm_output = model(
                input_ids=last, 
                past_key_values=perturbed_past
            )
            hidden_states_key = "hidden_states"
            
        all_logits, all_hidden = lm_output["logits"], lm_output[hidden_states_key]
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = nn.functional.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            #print(" pplm_bow_loss:", loss.data.cpu().numpy())

        if loss_type == 2 or loss_type == 3:
            ce_loss = nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            # TODO: check embedding 
            wte = model.get_input_embeddings()
            cls_wte = classifier.get_input_embeddings()
            output_so_far_input_embeds = model.model.shared(output_so_far)
            cls_output_so_far_input_embeds = classifier.roberta.embeddings(output_so_far)

            # for _ in range(horizon_length):
            #     inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
            #     if is_encoder_decoder: 
            #         lm_output = model(
            #             past_key_values=curr_unpert_past, 
            #             decoder_inputs_embeds=inputs_embeds,
            #             encoder_outputs=encoder_outputs
            #         )
            #     else:
            #         lm_output = model(
            #             past_key_values=curr_unpert_past, 
            #             inputs_embeds=inputs_embeds
            #         )

            #     curr_all_logits, curr_unpert_past, curr_all_hidden = (
            #         lm_output["logits"],
            #         lm_output["past_key_values"],
            #         lm_output[hidden_states_key],
            #     )
            #     curr_logits = curr_all_logits[:, -1, :]
            #     curr_probs = nn.functional.softmax(curr_logits, dim=-1)
            #     curr_probs = torch.unsqueeze(curr_probs, dim=1)
            #     curr_hidden = curr_all_hidden[-1]
            #     #new_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)
            #     new_accumulated_hidden = new_accumulated_hidden + curr_hidden[:, 0, :]

            cls_input_embeds = torch.matmul(curr_probs, cls_wte.weight.data)
            #new_accumulated_hidden = linear_convert_layer(new_accumulated_hidden)
            prediction = classification_head(
                inputs_embeds=torch.cat((cls_output_so_far_input_embeds, cls_input_embeds), 1), 
                features=new_accumulated_hidden / (curr_length + 1 + horizon_length), 
                model=classifier,
                mode="use_input_embeds"
            )

            label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            #print(prediction, label)
            #print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = nn.functional.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * ((corrected_probs * (corrected_probs / unpert_probs).log()).sum())
            #print(" kl_loss", kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        #print(" pplm_loss", (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index] ** gamma).data #.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append([layer_p_.detach() for layer_p_ in p_])
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [p_.requires_grad_(True).to(device=device) for p_ in grad_accumulator]
    if is_encoder_decoder: 
        grad_accumulator_split = [torch.split(layer_grad_accumulator, key_value_lens, 2) for layer_grad_accumulator in grad_accumulator]
        pert_past = [tuple(map(add, past[layer_id], grad_accumulator_split[layer_id])) for layer_id in range(len(grad_accumulator_split))]
    else:
        pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter

#TODO: get back classifier func
# def get_classifier(
#     name: Optional[str], class_label: Union[str, int], device: str
# ) -> Tuple[Optional[ClassificationHead], Optional[int]]:
#     if name is None:
#         return None, None

#     params = DISCRIMINATOR_MODELS_PARAMS[name]
#     classifier = ClassificationHead(class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
#     if "url" in params:
#         resolved_archive_file = cached_path(params["url"])
#     elif "path" in params:
#         resolved_archive_file = params["path"]
#     else:
#         raise ValueError("Either url or path have to be specified in the discriminator model parameters")
#     classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
#     classifier.eval()

#     if isinstance(class_label, str):
#         if class_label in params["class_vocab"]:
#             label_id = params["class_vocab"][class_label]
#         else:
#             label_id = params["default_class"]
#             print("class_label {} not in class_vocab".format(class_label))
#             print("available values are: {}".format(params["class_vocab"]))
#             print("using default class {}".format(label_id))

#     elif isinstance(class_label, int):
#         if class_label in set(params["class_vocab"].values()):
#             label_id = class_label
#         else:
#             label_id = params["default_class"]
#             print("class_label {} not in class_vocab".format(class_label))
#             print("available values are: {}".format(params["class_vocab"]))
#             print("using default class {}".format(label_id))

#     else:
#         label_id = params["default_class"]

#     return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append([tokenizer.encode(word.strip(), add_prefix_space=True) for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device="cuda"):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
    model,
    tokenizer,
    input_context=None,
    output_context=None,
    num_samples=1,
    device="cuda",
    bag_of_words=None,
    discrim=None,
    class_label=None,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
    base_gen=False,
    **kwargs
):

    if discrim in DISCRIMINATOR_MODELS_PARAMS: 
        classifier, class_id = get_classifier(discrim, class_label, device)
    else: 
        classifier, class_id = AutoModelForSequenceClassification.from_pretrained(discrim), class_label
    
    # classifier = classifier.half()
    classifier.to(device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"), tokenizer)

    if bag_of_words and classifier:
        #print("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")
        loss_type = PPLM_BOW_DISCRIM

    elif bag_of_words:
        loss_type = PPLM_BOW
        #print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        #print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    base_gen_time = 0
    
    if base_gen: 
        start_time = time.time()
        unpert_gen_tok_text, _, _ = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            input_context=input_context,
            output_context=output_context,
            device=device,
            length=length,
            sample=sample,
            perturb=False,
            repetition_penalty=repetition_penalty,
        )
        end_time = time.time()
        base_gen_time = end_time - start_time 
    else: 
        unpert_gen_tok_text = None 
    
    if device == "cuda":
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []
    model_gen_time = []

    for i in range(num_samples):
        start_time = time.time()
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            input_context=input_context,
            output_context=output_context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            repetition_penalty=repetition_penalty,
        )
        end_time = time.time()
        model_gen_time.append(end_time - start_time)
        pert_gen_tok_texts.append(pert_gen_tok_text)
        # if classifier is not None:
        #     discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == "cuda":
        torch.cuda.empty_cache()

    model_gen_time = sum(model_gen_time) / len(model_gen_time)

    # test accuracy 
    # unpert_logits = classifier(input_ids=unpert_gen_tok_text)
    # unpert_class_scores = nn.functional.softmax(unpert_logits["logits"], dim=-1)
    # unpert_class_score = unpert_class_scores[0, class_id]
    # pert_logits = classifier(input_ids=pert_gen_tok_text)
    # pert_class_scores = nn.functional.softmax(pert_logits["logits"], dim=-1)
    # pert_class_score = pert_class_scores[0, class_id]

    # print(f"unpert score {unpert_class_score}", unpert_class_scores, class_id)
    # print(f"pert_class_score {pert_class_score}", pert_class_scores, class_id)

    return unpert_gen_tok_text, pert_gen_tok_texts, \
            discrim_losses, losses_in_time, base_gen_time, model_gen_time

#TODO: fix input/output_context 
def generate_text_pplm(
    model,
    tokenizer,
    input_context=None,
    output_context=None,
    past=None,
    device="cuda",
    perturb=True,
    bow_indices=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
    is_encoder_decoder=True
):
    
    output_so_far = None
    
    if is_encoder_decoder:
        input_ids = torch.tensor(input_context, device=device, dtype=torch.long)
        input_ids = input_ids.unsqueeze(0)
        encoder_outputs = model.model.encoder(
            input_ids=input_ids
        )

    if output_context:
        output_context_t = torch.tensor(output_context, device=device, dtype=torch.long)
        while len(output_context_t.shape) < 2:
            output_context_t = output_context_t.unsqueeze(0)
        output_so_far = output_context_t

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []
    #for i in trange(length, ascii=True):
    for i in range(length):
        #print(f"step {i}")
        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                if is_encoder_decoder: 
                    past = model(
                        decoder_input_ids=output_so_far[:, :-1],
                        encoder_outputs=encoder_outputs
                    )["past_key_values"]
                else: 
                    past = model(output_so_far[:, :-1])["past_key_values"]

        if is_encoder_decoder: 
            lm_output = model(
                decoder_input_ids=output_so_far,
                encoder_outputs=encoder_outputs,
                return_dict=True
            )
            hidden_states_key = "decoder_hidden_states"
        else: 
            lm_output = model(
                input_ids=output_so_far,
                return_dict=True
            )
            hidden_states_key = "hidden_states"

        unpert_logits, unpert_past, unpert_all_hidden = (
            lm_output["logits"],
            lm_output["past_key_values"],
            lm_output[hidden_states_key],
        )
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    encoder_outputs=encoder_outputs,
                    output_so_far=output_so_far
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        # if i < 10: 
        #     for layer1 in pert_past: 
        #         print("layer1 length", len(layer1))
        #         for layer2 in layer1: 
        #             print(layer2.shape)
        # else: 
        #     raise ValueError("")

        if is_encoder_decoder: 
            lm_output = model(
                decoder_input_ids=last,
                encoder_outputs=encoder_outputs,
                past_key_values=pert_past
            )
        else: 
            lm_output = model(
                input_ids=last, 
                past_key_values=pert_past
            )

        pert_logits, past = (
            lm_output["logits"],
            lm_output["past_key_values"],
        )
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST

        #print(len(output_so_far[0]))
        #print(pert_logits.shape)
        # previous_tokens = sorted(list(set(output_so_far[0].tolist())))
        # repetition_mask = torch.ones_like(pert_logits[0, :])
        # repetition_mask[previous_tokens] = repetition_penalty
        # repetition_mask[pert_logits[0,:] >= 0] = 1 / repetition_mask[pert_logits[0,:] >= 0]
        # pert_logits[0, :] *= repetition_mask

        for token_idx in set(output_so_far[0].tolist()):
            if pert_logits[0, token_idx] < 0:
                pert_logits[0, token_idx] *= repetition_penalty
            else:
                pert_logits[0, token_idx] /= repetition_penalty

        pert_probs = nn.functional.softmax(pert_logits, dim=-1)

        # if classifier is not None:
        #     ce_loss = nn.CrossEntropyLoss()
        #     #prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
        #     prediction = classification_head(torch.mean(unpert_last_hidden, dim=1), classifier)
        #     label = torch.tensor([class_label], device=device, dtype=torch.long)
        #     unpert_discrim_loss = ce_loss(prediction, label)
        #     #print("unperturbed discrim loss", unpert_discrim_loss.data.cpu().numpy())
        # else:
        #     unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = nn.functional.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs**gm_scale) * (unpert_probs ** (1 - gm_scale))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = nn.functional.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        if last not in [tokenizer.eos_token_id, tokenizer.pad_token_id]: 
            output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        else: 
            break 

    return output_so_far, unpert_discrim_loss, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError("When using a generic discriminator, discrim_weights need to be specified")
    if discrim_meta is None:
        raise ValueError("When using a generic discriminator, discrim_meta need to be specified")

    with open(discrim_meta, "r") as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta["path"] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS["generic"] = meta


def run_pplm_example(
    pretrained_model="gpt2-medium",
    cond_text="",
    uncond=False,
    num_samples=1,
    bag_of_words=None,
    discrim=None,
    discrim_weights=None,
    discrim_meta=None,
    class_label=-1,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    seed=0,
    no_cuda=False,
    colorama=False,
    repetition_penalty=1.0,
    test_dta="",
    test_col="",
    test_num=1,
    output_dir=".",
    base_gen=False,
    log_every=2,
    max_tok_length=512,
    test_id=-1,
    streaming_out=False
):
    os.makedirs(output_dir, exist_ok=True)
    if streaming_out:
        streaming_out_f = open(os.path.join(output_dir, "s_out.txt"), "w", encoding="UTF-8")
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == "generic":
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None and discrim in DISCRIMINATOR_MODELS_PARAMS:
        pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim]["pretrained_model"]
        print("discrim = {}, pretrained_model set to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    #model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
    # model = model.half()
    model.to(device)
    model.eval()

    # load tokenizer
    #tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    test_col = test_col.split(",")
    if "src" not in test_col: 
        raise ValueError("The test data not include src column")
    
    test_dta = pd.read_csv(test_dta, sep="\t", names=test_col)
    model_generated_texts = []
    base_generated_texts = []
    if test_id >= 0:
        test_dta = test_dta.iloc[[test_id],:]
    else: 
        test_dta = test_dta.iloc[:test_num] 
        
    test_dta_num = len(test_dta)
    base_time_usage = []
    model_time_usage = []
    model_gen_class_labels = []

    gen_cache = {}

    for i in range(test_dta_num): 
        src = test_dta["src"].iloc[i]
        if class_label == -1: 
            model_gen_class_label = test_dta["id"].iloc[i]
        else:
            model_gen_class_label = class_label

        if src not in gen_cache: 
            input_context = tokenizer.encode(src, max_length=max_tok_length, truncation=True)
            output_context = tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)

            unpert_gen_tok_text, pert_gen_tok_texts, _, _, base_gen_time, model_gen_time = full_text_generation(
                model=model,
                tokenizer=tokenizer,
                input_context=input_context,
                output_context=output_context,
                device=device,
                num_samples=num_samples,
                bag_of_words=bag_of_words,
                discrim=discrim,
                class_label=model_gen_class_label,
                length=length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl_scale,
                repetition_penalty=repetition_penalty,
                base_gen=base_gen
            )

            if i % log_every == 0: 
                base_time_usage.append(base_gen_time)
                base_time_avg = sum(base_time_usage) / len(base_time_usage)
                model_time_usage.append(model_gen_time)
                model_time_avg = sum(model_time_usage) / len(model_time_usage)

                print(f"{i+1} sample is generated, base gen takes {base_time_avg}, model gen takes {model_time_avg}")
        else: 
            print(f"skip data {i}")


        if src not in gen_cache: 
            model_gen_txt = tokenizer.decode(pert_gen_tok_texts[0][0], skip_special_tokens=True)
            model_generated_texts.append(model_gen_txt)
            
            if base_gen: 
                base_gen_txt = tokenizer.decode(unpert_gen_tok_text[0], skip_special_tokens=True)
                base_generated_texts.append(base_gen_txt)
            else: 
                base_gen_txt = None 

            gen_cache[src] = (model_gen_txt, base_gen_txt)
        else: 
            model_generated_texts.append(gen_cache[src][0])
            if base_gen: 
                base_generated_texts.append(gen_cache[src][1])

        model_gen_class_labels.append(model_gen_class_label)

        if streaming_out:
            orig_cols = list(map(str, test_dta.iloc[i, :]))
            streaming_out_f.write("\t".join(orig_cols + [model_generated_texts[-1]]) + "\n")

        if i % 10 == 0:
            print(f"{i}, cat: {model_gen_class_label}, {model_generated_texts[-1]}")
        
    test_dta["model_gen"] = model_generated_texts
    test_dta["model_gen_label"] = model_gen_class_labels
    
    # bug to fix for basegen 
    if base_gen: 
        test_dta["base_gen"] = base_generated_texts
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "predict.txt")
    test_dta.to_csv(out_path, sep="\t", index=False, header=False, quoting=3)
    
    return

import pandas as pd 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument("--cond_text", type=str, default="The lake", help="Prefix texts to condition on")
    parser.add_argument("--uncond", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help=(
            "Bags of words used for PPLM-BoW. "
            "Either a BOW id (see list in code) or a filepath. "
            "Multiple BoWs separated by ;"
        ),
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        #choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument(
        "--discrim_weights",
        type=str,
        default=None,
        help="Weights for the generic discriminator",
    )
    parser.add_argument(
        "--discrim_meta",
        type=str,
        default=None,
        help="Meta information for the generic discriminator",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; 0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true", help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true", help="colors keywords")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Penalize repetition. More than 1.0 -> less repetition",
    )

    parser.add_argument("--test_dta", type=str)
    parser.add_argument("--test_col", type=str)
    parser.add_argument("--test_num", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--base_gen", action="store_true")
    parser.add_argument("--log_every", type=int, default=2)
    parser.add_argument("--max_tok_length", type=int, default=512)
    parser.add_argument("--test_id", type=int, default=-1)
    parser.add_argument("--streaming_out", action="store_true", default=False)

    args = parser.parse_args()
    run_pplm_example(**vars(args))
