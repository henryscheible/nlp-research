import torch
from huggingface_hub import HfApi
from nlpcore.bias_datasets.stereoset import load_stereoset, process_stereoset
from nlpcore.bert.bert import load_bert_model
from nlpcore.train import train_model

print(f"=======IS CUDA AVAILABLE: {torch.cuda.is_available()}==========")

tokenizer, model = load_bert_model(num_labels=3)
raw_dataset = load_stereoset()
train, eval = process_stereoset(raw_dataset, tokenizer, include_unrelated=True)
# Create train and eval dataloaders
best_model, validation = train_model(
    model,
    model.parameters(),
    train,
    eval,
    epochs=3
)
best_model.push_to_hub("stereoset_all_bert_all")
tokenizer.push_to_hub("stereoset_all_bert_all")
api = HfApi()
api.upload_file(
    path_or_fileobj="out/validation.json",
    path_in_repo="validation.json",
    repo_id="henryscheible/stereoset_all_bert_all",
    repo_type="model",
)
print(validation)
