import os

import numpy as np
import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.stereoset import load_processed_stereoset
from nlpcore.bias_datasets.winobias import load_processed_winobias
from nlpcore.bias_datasets.crows_pairs import load_processed_crows_pairs
from nlpcore.shapley import get_shapley
from transformers import AutoTokenizer

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

DATASET = os.environ.get("DATASET")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES"))

CHECKPOINT = os.environ.get("CHECKPOINT")
REPO = "henryscheible/"+CHECKPOINT

print(f"=======CHECKPOINT: {CHECKPOINT}==========")

tokenizer = AutoTokenizer.from_pretrained(REPO)
if DATASET == "stereoset":
    _, eval = load_processed_stereoset(tokenizer, include_unrelated=False)
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
