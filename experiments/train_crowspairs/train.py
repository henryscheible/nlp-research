import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.crows_pairs import load_processed_crows_pairs
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model
import os

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

FINETUNE = os.environ.get("FINETUNE")

train_tag = "finetuned" if FINETUNE == "True" else "classifieronly"
checkpoint = f"crows_pairs_bert_{train_tag}"

print(f"Model Checkpoint Name: {checkpoint}")

tokenizer, model = load_bert_model()
train, eval = load_processed_crows_pairs(tokenizer)
# Create train and eval dataloaders
param_dict = dict(model.named_parameters())

params = model.parameters() if FINETUNE else [
    param_dict["classifier.weight"],
    param_dict["classifier.bias"]
]
best_model, validation = train_model(
    model,
    params,
    train,
    eval,
)
best_model.push_to_hub(f"henryscheible/{checkpoint}")
tokenizer.push_to_hub(f"henryscheible/{checkpoint}")
api = HfApi()
api.upload_file(
    path_or_fileobj="out/validation.json",
    path_in_repo="validation.json",
    repo_id=f"henryscheible/{checkpoint}",
    repo_type="model",
)
print(validation)
