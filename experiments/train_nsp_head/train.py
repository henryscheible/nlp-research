import os

import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.stereoset import load_processed_stereoset
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model
import transformers

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

INCLUDE_UNRELATED = os.environ.get("INCLUDE_UNRELATED")
FINETUNE = os.environ.get("FINETUNE")

data_tag = "all" if INCLUDE_UNRELATED == "True" else "binary"
train_tag = "finetuned" if FINETUNE == "True" else "classifieronly"
checkpoint = f"stereoset_{data_tag}_bert_{train_tag}"

print(f"Model Configuration: INCLUDE_UNRELATED: {INCLUDE_UNRELATED}, FINETUNE: {FINETUNE}")
print(f"Model Checkpoint Name: {checkpoint}")

num_labels = 3 if INCLUDE_UNRELATED == "True" else 2

tokenizer, model = load_bert_model(num_labels=num_labels)
train, eval = load_processed_stereoset(tokenizer, include_unrelated=(INCLUDE_UNRELATED == "True"))
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
    epochs=6,
)
api = HfApi()
try:
    api.create_repo(
        repo_id=f"henryscheible/{checkpoint}"
    )
except:
    pass
api.upload_file(
    path_or_fileobj=f"out/validation.json",
    path_in_repo=f"validation.json",
    repo_id=f"henryscheible/{checkpoint}",
    repo_type="model",
)
try:
    os.remove("./out/validation.json")
except:
    pass
best_model.push_to_hub(f"{checkpoint}")
tokenizer.push_to_hub(f"{checkpoint}")
print(validation)
