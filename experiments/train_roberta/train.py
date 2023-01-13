import os

import evaluate
import numpy as np
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, \
    get_scheduler, Trainer
from torch.optim import AdamW

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

FINETUNE = os.environ.get("FINETUNE")

MODEL = os.environ.get("MODEL")
DATASET = os.environ.get("DATASET")


train_tag = "finetuned" if FINETUNE == "True" else "classifieronly"
checkpoint = f"{DATASET}_trainer_{MODEL}_{train_tag}"

print(f"Model Checkpoint Name: {checkpoint}")

num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if DATASET == "stereoset":
    raw_dataset = load_dataset("stereoset", "intersentence")['validation']
    dataset = process_stereoset(raw_dataset, tokenizer)
elif DATASET == "winobias":
    dataset = process_winobias(tokenizer)
elif DATASET == "crowspairs":
    raw_dataset = load_dataset("crows_pairs")['test']
    dataset = process_crows_pairs(raw_dataset, tokenizer)

# Create train and eval dataloaders
param_dict = dict(model.named_parameters())
if "roberta" in MODEL:
    param_names = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
elif "bert" in MODEL:
    param_names = ["classifier.weight", "classifier.bias"]
print(f"PARAM NAMES: {param_names}")
params = model.parameters() if FINETUNE == "True" else [param_dict[name] for name in param_names]
num_epochs = int(os.environ.get("EPOCHS"))

training_args = TrainingArguments(checkpoint,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    save_steps=50,
    eval_steps=50,
    push_to_hub=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    hub_strategy="every_save",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32
)

train_loader = DataLoader(dataset["train"], batch_size=32)
num_training_steps = num_epochs * len(train_loader)

optimizer = AdamW(params, lr=5e-5)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1),
    return metric.compute(predictions=predictions[0], references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()