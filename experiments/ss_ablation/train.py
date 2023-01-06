import json
import os

import evaluate
import numpy as np
import pandas as pd
import requests
import json

import evaluate
import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import HfApi
from nlpcore.stereotypescore import StereotypeScoreCalculator
from nlpcore.maskmodel import MaskModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction
import torch
from datetime import datetime

MODEL = os.environ.get("MODEL")

def pull_contribs(checkpoint, suffix):
    res = requests.get(f"https://huggingface.co/henryscheible/{checkpoint}/raw/main/output-contribs-{suffix}.txt")
    print(res.text)
    return json.loads(res.text)


def get_positive_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution > 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu")


def get_negative_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution < 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu")


def get_bottom_up_masks(contribs):
    sorted_indices = np.argsort(contribs)
    masks = [np.zeros(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 1
        masks += [new_mask]
    return [torch.tensor(mask).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def get_top_down_masks(contribs):
    sorted_indices = np.argsort(contribs)
    masks = [np.ones(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 0
        masks += [new_mask]
    return [torch.tensor(mask).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]

def get_bottom_up_masks_rev(contribs):
    sorted_indices = np.argsort(-np.array(contribs))
    masks = [np.zeros(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 1
        masks += [new_mask]
    return [torch.tensor(mask).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def get_top_down_masks_rev(contribs):
    sorted_indices = np.argsort(-np.array(contribs))
    masks = [np.ones(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 0
        masks += [new_mask]
    return [torch.tensor(mask).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def get_ss(calc, inner_model, mask=None):
    inner_model.eval()
    inner_model.to("cuda" if torch.cuda.is_available() else "cpu")
    mask = torch.ones((12, 12)).to("cuda" if torch.cuda.is_available() else "cpu") if mask is None else mask
    model = MaskModel(inner_model, mask).to("cuda" if torch.cuda.is_available() else "cpu")
    calc.set_intersentence_model(model)
    return calc()


def test_shapley(checkpoint, suffix):
    REPO = "henryscheible/" + checkpoint
    print(f"=======CHECKPOINT: {checkpoint}==========")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForNextSentencePrediction.from_pretrained(MODEL)
    calc = StereotypeScoreCalculator(model, tokenizer, model, tokenizer)
    base_ss, base_lm = get_ss(calc, model)


    contribs = pull_contribs(checkpoint, suffix)

    print("CALCULATING BOTTOM_UP")

    progress_bar = tqdm(range(144))

    bottom_up_ss = []
    bottom_up_lm = []
    for mask in get_bottom_up_masks(contribs):
        ss, lm = get_ss(calc, model, mask=mask)
        bottom_up_ss += [ss]
        bottom_up_lm += [lm]
        progress_bar.update(1)

    print("CALCULATING TOP_DOWN")
    #
    #
    # progress_bar = tqdm(range(144))
    #
    # top_down_ss = []
    # top_down_lm = []
    # for mask in get_top_down_masks(contribs):
    #     ss, lm = get_ss(calc, model, mask=mask)
    #     top_down_ss += [ss]
    #     top_down_lm += [ss]
    #     progress_bar.update(1)
    #
    # print("CALCULATING BOTTOM_UP_REV")
    #
    # progress_bar = tqdm(range(144))
    #
    #
    # bottom_up_rev_ss = []
    # bottom_up_rev_lm = []
    # for mask in get_bottom_up_masks_rev(contribs):
    #     ss, lm = get_ss(calc, model, mask=mask)
    #     bottom_up_rev_ss += [ss]
    #     bottom_up_rev_lm += [ss]
    #     progress_bar.update(1)
    #
    # print("CALCULATING TOP_DOWN_REV")
    #
    #
    # progress_bar = tqdm(range(144))
    #
    # top_down_rev_ss = []
    # top_down_rev_lm = []
    # for mask in get_top_down_masks_rev(contribs):
    #     ss, lm = get_ss(calc, model, mask=mask)
    #     top_down_rev_ss += [ss]
    #     top_down_rev_lm += [ss]
    #     progress_bar.update(1)

    return {
        "base_ss": base_ss,
        "base_lm": base_lm,
        "contribs": contribs,
        "bottom_up_ss": list(bottom_up_ss),
        "bottom_up_lm": list(bottom_up_lm),
        # "top_down_ss": list(top_down_ss),
        # "top_down_lm": list(top_down_lm),
        # "bottom_up_rev_ss": list(bottom_up_rev_ss),
        # "bottom_up_rev_lm": list(bottom_up_rev_lm),
        # "top_down_rev_ss": list(top_down_rev_ss),
        # "top_down_rev_lm": list(top_down_rev_lm),
    }


checkpoints = [
    "stereoset_binary_bert_classifieronly",
]

suffixes = [
    "250",
    # "750",
    # "500",
    # "1000"
]


def get_results():
    ret = dict()
    for checkpoint in checkpoints:
        checkpoint_results = {}
        for suffix in suffixes:
            checkpoint_results[str(suffix)] = test_shapley(checkpoint, suffix)
        ret[checkpoint] = checkpoint_results
    return ret


results = get_results()

print(results)

with open("results.json", "a") as file:
    file.write(json.dumps(results))

time = datetime.now()
api = HfApi()
api.upload_file(
    path_or_fileobj="results.json",
    path_in_repo=f"results_{time}.json",
    repo_id=f"henryscheible/experiment_results",
    repo_type="model",
)
