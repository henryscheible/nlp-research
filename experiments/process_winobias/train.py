import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.winobias import load_winobias, process_winobias
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model

tokenizer, model = load_bert_model()
raw_dataset = load_winobias()
dataset = process_winobias(raw_dataset, tokenizer)
dataset.push_to_hub("henryscheible/winobias")

