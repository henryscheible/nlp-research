import json
import os
from collections import defaultdict

import evaluate
import numpy as np
import torch
import transformers
from captum.attr import ShapleyValueSampling
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
from nlpcore.bias_datasets.winobias import load_processed_winobias
from nlpcore.bias_datasets.crows_pairs import load_processed_crows_pairs
from nlpcore.shapley import get_shapley
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")
#
# DATASET = os.environ.get("DATASET")
# NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES"))
#
# CHECKPOINT = os.environ.get("CHECKPOINT")
DATASET = "stereoset"
NUM_SAMPLES = 250
CHECKPOINT = "stereoset_binary_roberta-base_classifieronly"
REPO = "henryscheible/"+CHECKPOINT

tokenizer = AutoTokenizer.from_pretrained(REPO)
if DATASET == "stereoset":
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    raw_dataset = load_dataset("stereoset", "intersentence")['validation']
    print(f"Downloaded Dataset: {raw_dataset}")
    dataset = process_stereoset(raw_dataset, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train = DataLoader(dataset["train"], shuffle=True, batch_size=64, collate_fn=data_collator)
    eval = DataLoader(dataset["eval"], batch_size=64, collate_fn=data_collator)
elif DATASET == "winobias":
    _, eval = load_processed_winobias(tokenizer)
elif DATASET == "crows_pairs":
    _, eval = load_processed_crows_pairs(tokenizer)


get_shapley(eval, REPO, num_samples=NUM_SAMPLES)
api = HfApi()
filename = f"contribs-{NUM_SAMPLES}.txt"
api.upload_file(
    path_or_fileobj=f"contribs.txt",
    path_in_repo=filename,
    repo_id=f"henryscheible/{CHECKPOINT}",
    repo_type="model",
)
