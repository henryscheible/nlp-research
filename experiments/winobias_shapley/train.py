import os

import numpy as np
import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.winobias import load_processed_winobias
from nlpcore.shapley import get_shapley
from transformers import AutoTokenizer

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

FINETUNE = os.environ.get("FINETUNE")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES"))
SEED = int(os.environ.get("SEED")) if "SEED" in os.environ.keys() else None
SUFFIX = os.environ.get("SUFFIX") if "SUFFIX" in os.environ.keys() else None

train_tag = "finetuned" if FINETUNE == "True" else "classifieronly"
print(f"suffix: {SUFFIX}")
CHECKPOINT = f"winobias_bert_{train_tag}_{SUFFIX}" if SUFFIX is not None and SUFFIX != "" else f"winobias_bert_{train_tag}"
REPO = "henryscheible/"+CHECKPOINT

print(f"=======CHECKPOINT: {CHECKPOINT}==========")

if SEED is not None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(REPO)
train, eval = load_processed_winobias(tokenizer)
get_shapley(eval, REPO, num_samples=NUM_SAMPLES)
api = HfApi()
filename = f"contribs-{NUM_SAMPLES}_{SEED}.txt" if SEED is not None else f"contribs-{NUM_SAMPLES}.txt"
api.upload_file(
    path_or_fileobj=f"contribs.txt",
    path_in_repo=filename,
    repo_id=f"henryscheible/{CHECKPOINT}",
    repo_type="model",
)
