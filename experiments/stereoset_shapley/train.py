import os

import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.stereoset import load_processed_stereoset
from nlpcore.shapley import get_shapley
from transformers import AutoTokenizer

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

INCLUDE_UNRELATED = os.environ.get("INCLUDE_UNRELATED")
FINETUNE = os.environ.get("FINETUNE")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES"))

data_tag = "all" if INCLUDE_UNRELATED == "True" else "binary"
train_tag = "finetuned" if FINETUNE == "True" else "classifieronly"
CHECKPOINT = f"stereoset_{data_tag}_bert_{train_tag}"
REPO = "henryscheible/"+CHECKPOINT

tokenizer = AutoTokenizer.from_pretrained(REPO)
train, eval = load_processed_stereoset(tokenizer, include_unrelated=(INCLUDE_UNRELATED == "True"))
get_shapley(eval, REPO, num_samples=NUM_SAMPLES)
api = HfApi()
api.upload_file(
    path_or_fileobj=f"contribs.txt",
    path_in_repo=f"contribs-{NUM_SAMPLES}.txt",
    repo_id=f"henryscheible/{CHECKPOINT}",
    repo_type="model",
)
