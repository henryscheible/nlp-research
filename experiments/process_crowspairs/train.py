import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.crows_pairs import load_crows_pairs, process_crows_pairs
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model

tokenizer, model = load_bert_model()
raw_dataset = load_crows_pairs()
dataset = process_crows_pairs(raw_dataset, tokenizer)
dataset.push_to_hub("henryscheible/crows_pairs")

