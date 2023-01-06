import os

import torch
from huggingface_hub import HfApi
from datasets import load_dataset, DatasetDict
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

def process_fn_binary(example):
    sentences = []
    labels = []
    contexts = []
    for i in range(3):
        print(f"example: {example}")
        if example["sentences"][0]["gold_label"][i] != 2:
            sentences.append(example["sentences"][0]["sentence"][i])
            labels.append(example["sentences"][0]["gold_label"][i])
            contexts.append(example["context"][0])
    return {
        "sentence": sentences,
        "label": labels,
        "context": contexts
    }


def process_fn_all(example):
    sentences = []
    labels = []
    contexts = []
    for i in range(3):
        sentences.append(example["sentences"][0]["sentence"][i])
        labels.append(example["sentences"][0]["gold_label"][i])
        contexts.append(example["context"][0])
    return {
        "sentence": sentences,
        "label": labels,
        "context": contexts
    }


def process_stereoset(dataset, tokenizer, batch_size=64, include_unrelated=False):
    def tokenize(example):
        return tokenizer(example["context"], example["sentence"], truncation=True, padding=True)

    num_samples = len(dataset["id"])
    dataset = dataset.remove_columns([
        "id",
        "target",
        "bias_type",
    ])
    process_fn = process_fn_all if include_unrelated else process_fn_binary
    dataset_processed = dataset.map(process_fn, batched=True, batch_size=1, remove_columns=["sentences"])
    print(dataset_processed.column_names)
    tokenized_dataset = dataset_processed.map(tokenize, batched=True, batch_size=64,
                                              remove_columns=["context", "sentence"])
    print(tokenized_dataset.column_names)

    split_tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=0.3
    )

    return DatasetDict({
        "train": split_tokenized_dataset["train"],
        "eval": split_tokenized_dataset["test"]
    })

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

INCLUDE_UNRELATED = os.environ.get("INCLUDE_UNRELATED")
FINETUNE = os.environ.get("FINETUNE")

MODEL = os.environ.get("MODEL")

data_tag = "all" if INCLUDE_UNRELATED == "True" else "binary"
train_tag = "finetuned" if FINETUNE == "True" else "classifieronly"
checkpoint = f"stereoset_{data_tag}_{MODEL}_{train_tag}"

print(f"Model Configuration: INCLUDE_UNRELATED: {INCLUDE_UNRELATED}, FINETUNE: {FINETUNE}")
print(f"Model Checkpoint Name: {checkpoint}")

num_labels = 3 if INCLUDE_UNRELATED == "True" else 2

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
raw_dataset = load_dataset("stereoset", "intersentence")['validation']
print(f"Downloaded Dataset: {raw_dataset}")
dataset = process_stereoset(raw_dataset, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train = DataLoader(dataset["train"], shuffle=True, batch_size=64, collate_fn=data_collator)
eval = DataLoader(dataset["eval"], batch_size=64, collate_fn=data_collator)
# Create train and eval dataloaders
param_dict = dict(model.named_parameters())
if "roberta" in MODEL:
    param_names = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
elif "bert" in MODEL:
    param_names = ["classifier.weight", "classifier.bias"]

params = model.parameters() if FINETUNE else [param_dict[name] for name in param_names]
best_model, validation = train_model(
    model,
    params,
    train,
    eval,
    epochs=10,
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
