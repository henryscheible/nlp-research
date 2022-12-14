import json

import evaluate
import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.winobias import load_processed_winobias
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import matplotlib.pyplot as plt


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


def evaluate_model(eval_loader, model, mask=None):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    metric = evaluate.load('accuracy')

    for eval_batch in eval_loader:
        eval_batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in eval_batch.items()}
        with torch.no_grad():
            outputs = model(**eval_batch, head_mask=mask) if mask is not None else model(**eval_batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=eval_batch["labels"])

    return float(metric.compute()["accuracy"])


def test_shapley(checkpoint, loader, suffix):
    REPO = "henryscheible/" + checkpoint
    print(f"=======CHECKPOINT: {checkpoint}==========")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    _, eval_loader = loader(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(REPO)
    base_acc = evaluate_model(eval_loader, model)

    contribs = pull_contribs(checkpoint, suffix)

    progress_bar = tqdm(range(144))

    bottom_up_results = []
    for mask in get_bottom_up_masks(contribs):
        bottom_up_results += [evaluate_model(eval_loader, model, mask=mask)]
        progress_bar.update(1)

    progress_bar = tqdm(range(144))

    top_down_results = []
    for mask in get_top_down_masks(contribs):
        top_down_results += [evaluate_model(eval_loader, model, mask=mask)]
        progress_bar.update(1)

    progress_bar = tqdm(range(144))

    bottom_up_rev_results = []
    for mask in get_bottom_up_masks_rev(contribs):
        bottom_up_rev_results += [evaluate_model(eval_loader, model, mask=mask)]
        progress_bar.update(1)

    progress_bar = tqdm(range(144))

    top_down_rev_results = []
    for mask in get_top_down_masks_rev(contribs):
        top_down_rev_results += [evaluate_model(eval_loader, model, mask=mask)]
        progress_bar.update(1)

    return {
        "base_acc": base_acc,
        "contribs": contribs,
        "bottom_up_results": list(bottom_up_results),
        "top_down_results": list(top_down_results),
        "bottom_up_rev_results": list(bottom_up_rev_results),
        "top_down_rev_results": list(top_down_rev_results)
    }


# def generate_plots(checkpoint, results):
#     # Bottom Up Plot
#     processed_data = pd.DataFrame({
#         'heads': np.arange(145),
#         'bottom_up': results["bottom_up_results"],
#         'top_down': results["top_down_results"]
#     })
#     plt.figure(dpi=300)
#     bottom_up_plot = sns.lineplot(x='heads', y='value', hue='variable',
#                                   data=pd.melt(processed_data, ['heads']))
#     bottom_up_plot.set(
#         title=f"{checkpoint}",
#         xlabel="# of Attention Heads Added",
#         ylabel="Accuracy"
#     )
#     plt.axvline(results["vline"], 0, 1)
#     plt.savefig(f'{checkpoint}-accuracy-test.png')
#
#     plt.figure(dpi=300)
#     heatmap = sns.heatmap(np.array(pull_contribs(checkpoint)).reshape((12, 12)), cmap="GiR")
#     heatmap.set(
#         title=f"{checkpoint} Attention Head Contributions"
#     )
#     plt.savefig(f'{checkpoint}-heatmap.png')
#     api = HfApi()
#     api.upload_file(
#         path_or_fileobj=f'{checkpoint}-accuracy-test.png',
#         path_in_repo=f'{checkpoint}-accuracy-test.png',
#         repo_id=f"henryscheible/{checkpoint}",
#         repo_type="model",
#     )
#     api.upload_file(
#         path_or_fileobj=f'{checkpoint}-heatmap.png',
#         path_in_repo=f'{checkpoint}-heatmap.png',
#         repo_id=f"henryscheible/{checkpoint}",
#         repo_type="model",
#     )


checkpoints = [
    ("winobias_bert_classifieronly", load_processed_winobias),
]

suffixes = [
    "10",
    # "750",
    # "500",
    # "1000"
]


def get_results():
    ret = dict()
    for checkpoint in checkpoints:
        checkpoint_results = {}
        for suffix in suffixes:
            checkpoint_results[str(suffix)] = test_shapley(*checkpoint, suffix)
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
