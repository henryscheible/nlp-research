import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.stereoset import load_stereoset, process_stereoset
from nlpcore.shapley import get_shapley
from transformers import AutoModel, AutoTokenizer

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")
CHECKPOINT = "henryscheible/stereoset_binary_bert_predheadonly"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
raw_dataset = load_stereoset()
train, eval = process_stereoset(raw_dataset, tokenizer)
get_shapley(eval, CHECKPOINT, num_samples=30)
api = HfApi()
api.upload_file(
    path_or_fileobj="contribs.txt",
    path_in_repo="contribs.txt",
    repo_id="henryscheible/stereoset_binary_bert_predheadonly",
    repo_type="model",
)
