import os

import torch
from huggingface_hub import HfApi
from datasets import load_dataset, DatasetDict
from nlpcore.bert.bert import load_bert_model
from nlpcore.bias_datasets.stereoset import process_stereoset
from nlpcore.bias_datasets.winobias import process_winobias
from nlpcore.bias_datasets.crows_pairs import process_crows_pairs
from nlpcore.train import train_model
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

INCLUDE_UNRELATED = os.environ.get("INCLUDE_UNRELATED")
FINETUNE = os.environ.get("FINETUNE")

MODEL = os.environ.get("MODEL")
DATASET = os.environ.get("DATASET")


train_tag = "finetuned" if FINETUNE == "True" else "classifieronly"
checkpoint = f"{DATASET}_{MODEL}_{train_tag}"

print(f"Model Configuration: INCLUDE_UNRELATED: {INCLUDE_UNRELATED}, FINETUNE: {FINETUNE}")
print(f"Model Checkpoint Name: {checkpoint}")

num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if DATASET == "stereoset":
    raw_dataset = load_dataset("stereoset", "intersentence")['validation']
    print(f"Downloaded Dataset: {raw_dataset}")
    dataset = process_stereoset(raw_dataset, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train = DataLoader(dataset["train"], shuffle=True, batch_size=24, collate_fn=data_collator)
    eval = DataLoader(dataset["eval"], batch_size=64, collate_fn=data_collator)
elif DATASET == "winobias":
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = process_winobias(tokenizer)
    train = DataLoader(dataset["train"], shuffle=True, batch_size=24, collate_fn=data_collator)
    eval = DataLoader(dataset["eval"], batch_size=64, collate_fn=data_collator)
elif DATASET == "crowspairs":
    raw_dataset = load_dataset("crows_pairs")['test']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = process_crows_pairs(raw_dataset, tokenizer)
    train = DataLoader(dataset["train"], shuffle=True, batch_size=24, collate_fn=data_collator)
    eval = DataLoader(dataset["test"], batch_size=64, collate_fn=data_collator)

# Create train and eval dataloaders
param_dict = dict(model.named_parameters())
if "roberta" in MODEL:
    param_names = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
elif "bert" in MODEL:
    param_names = ["classifier.weight", "classifier.bias"]
print(f"PARAM NAMES: {param_names}")
params = model.parameters() if FINETUNE == "True" else [param_dict[name] for name in param_names]
best_model, validation = train_model(
    model,
    params,
    train,
    eval,
    epochs=12,
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
