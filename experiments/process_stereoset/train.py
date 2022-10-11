import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.stereoset import load_stereoset, process_stereoset
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model

tokenizer, model = load_bert_model()
raw_dataset = load_stereoset()
dataset_binary = process_stereoset(raw_dataset, tokenizer)
dataset_all = process_stereoset(raw_dataset, tokenizer, include_unrelated=True)
dataset_binary.push_to_hub("henryscheible/stereoset_binary")
dataset_all.push_to_hub("henryscheible/stereoset_all")

