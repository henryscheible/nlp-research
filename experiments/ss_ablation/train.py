import json

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
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import datetime




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


def get_ss(inner_model, tokenizer, mask=None):
    inner_model.eval()
    inner_model.to("cuda" if torch.cuda.is_available() else "cpu")
    mask = torch.ones((12, 12)).to("cuda" if torch.cuda.is_available() else "cpu") if mask is None else mask

    model = MaskModel(inner_model, mask).to("cuda" if torch.cuda.is_available() else "cpu")
    calc = StereotypeScoreCalculator(model, tokenizer, model, tokenizer)
    print(f"MODEL: {model}")
    print(f"MASK: {mask}")


    return calc()


def test_shapley(checkpoint, suffix):
    REPO = "henryscheible/" + checkpoint
    print(f"=======CHECKPOINT: {checkpoint}==========")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    base_ss = get_ss(model, tokenizer)


    contribs = pull_contribs(checkpoint, suffix)

    progress_bar = tqdm(range(144))

    bottom_up_results = []
    for mask in get_bottom_up_masks(contribs):
        bottom_up_results += [get_ss(model, tokenizer, mask=mask)]
        progress_bar.update(1)

    progress_bar = tqdm(range(144))

    top_down_results = []
    for mask in get_top_down_masks(contribs):
        top_down_results += [get_ss(model, tokenizer, mask=mask)]
        progress_bar.update(1)

    progress_bar = tqdm(range(144))

    bottom_up_rev_results = []
    for mask in get_bottom_up_masks_rev(contribs):
        bottom_up_rev_results += [get_ss(model, tokenizer, mask=mask)]
        progress_bar.update(1)

    progress_bar = tqdm(range(144))

    top_down_rev_results = []
    for mask in get_top_down_masks_rev(contribs):
        top_down_rev_results += [get_ss(model, tokenizer, mask=mask)]
        progress_bar.update(1)

    return {
        "base_acc": base_ss,
        "contribs": contribs,
        "bottom_up_results": list(bottom_up_results),
        "top_down_results": list(top_down_results),
        "bottom_up_rev_results": list(bottom_up_rev_results),
        "top_down_rev_results": list(top_down_rev_results)
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
        ret[checkpoint[0]] = checkpoint_results
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
