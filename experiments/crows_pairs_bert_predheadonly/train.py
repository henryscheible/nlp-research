import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.crows_pairs import load_crows_pairs, process_crows_pairs
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

tokenizer, model = load_bert_model()
raw_dataset = load_crows_pairs()
train, eval = process_crows_pairs(raw_dataset, tokenizer)
# Create train and eval dataloaders
paramdict = dict(model.named_parameters())
best_model, validation = train_model(
    model,
    [
        paramdict["classifier.weight"],
        paramdict["classifier.bias"]
    ],
    train,
    eval,
)
best_model.push_to_hub("crows_pairs_bert_predheadonly")
tokenizer.push_to_hub("crows_pairs_bert_predheadonly")
api = HfApi()
api.upload_file(
    path_or_fileobj="out/validation.json",
    path_in_repo="validation.json",
    repo_id="henryscheible/crows_pairs_bert_predheadonly",
    repo_type="model",
)
print(validation)
