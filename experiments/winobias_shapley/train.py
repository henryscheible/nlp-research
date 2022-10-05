import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.winobias import load_winobias, process_winobias
from nlpcore.shapley import get_shapley
from transformers import AutoModel, AutoTokenizer

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")
CHECKPOINT = "henryscheible/crows_pairs_bert_predheadonly"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
raw_dataset = load_winobias()
train, eval = process_winobias(raw_dataset, tokenizer)
get_shapley(eval, CHECKPOINT, num_samples=500, num_perturbations_per_eval=10)
api = HfApi()
api.upload_file(
    path_or_fileobj="out/contribs.txt",
    path_in_repo="contribs.txt",
    repo_id="henryscheible/crows_pairs_bert_predheadonly",
    repo_type="model",
)
