import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.crows_pairs import load_crows_pairs, process_crows_pairs
from nlpcore.shapley import get_shapley
from transformers import AutoModel, AutoTokenizer

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")
CHECKPOINT = "henryscheible/stereoset_all_bert_all"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
raw_dataset = load_crows_pairs()
train, eval = process_crows_pairs(raw_dataset, tokenizer, batch_size=4096)
get_shapley(eval, CHECKPOINT)
api = HfApi()
api.upload_file(
    path_or_fileobj="out/contribs.txt",
    path_in_repo="contribs.txt",
    repo_id="henryscheible/stereoset_all_bert_all",
    repo_type="model",
)
