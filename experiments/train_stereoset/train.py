import os

import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.stereoset import load_processed_stereoset
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

INCLUDE_UNRELATED = os.environ.get("INCLUDE_UNRELATED")
FINETUNE = os.environ.get("FINETUNE")

data_tag = "all" if INCLUDE_UNRELATED else "binary"
train_tag = "classifieronly" if FINETUNE else "finetuned"
checkpoint = f"stereoset_{data_tag}_bert_{train_tag}"

print(f"Model Configuration: INCLUDE_UNRELATED: {INCLUDE_UNRELATED}, FINETUNE: {FINETUNE}")
print(f"Model Checkpoint Name: {checkpoint}")

num_labels = 3 if INCLUDE_UNRELATED else 2

tokenizer, model = load_bert_model(num_labels=num_labels)
train, eval = load_processed_stereoset(tokenizer, include_unrelated=INCLUDE_UNRELATED)
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
    epochs=8
)
best_model.push_to_hub(checkpoint)
tokenizer.push_to_hub(checkpoint)
api = HfApi()
api.upload_file(
    path_or_fileobj="out/validation.json",
    path_in_repo="validation.json",
    repo_id=f"henryscheible/{checkpoint}",
    repo_type="model",
)
print(validation)
