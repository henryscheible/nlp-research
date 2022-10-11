import evaluate
import torch
from nlpcore.bias_datasets.stereoset import load_stereoset, process_stereoset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from get_summary import contribs


def get_positive_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution > 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).reshape(12, 12).to("cuda")


def get_negative_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution < 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).reshape(12, 12).to("cuda")


def evaluate_model(eval_loader, model, mask=None):
    model.eval()
    model.to("cuda")
    metric = evaluate.load('accuracy')

    progress_bar = tqdm(range(len(eval_loader)))

    for eval_batch in eval_loader:
        eval_batch = {k: v.to("cuda") for k, v in eval_batch.items()}
        with torch.no_grad():
            outputs = model(**eval_batch, head_mask=mask) if mask is not None else model(**eval_batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=eval_batch["labels"])
        progress_bar.update(1)

    return metric.compute()


def test_shapley(checkpoint, include_unrelated):
    REPO = "henryscheible/" + checkpoint
    print(f"=======CHECKPOINT: {checkpoint}==========")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    raw_dataset = load_stereoset()
    _, eval_loader = process_stereoset(raw_dataset, tokenizer, include_unrelated=include_unrelated)
    model = AutoModelForSequenceClassification.from_pretrained(REPO)
    base_acc = evaluate_model(eval_loader, model)

    pos_mask = get_positive_mask(contribs[checkpoint])
    pos_acc = evaluate_model(eval_loader, model, mask=pos_mask)

    neg_mask = get_negative_mask(contribs[checkpoint])
    neg_acc = evaluate_model(eval_loader, model, mask=neg_mask)

    return {
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
        "base_acc": base_acc,
        "pos_acc": pos_acc,
        "neg_acc": neg_acc
    }


checkpoints = [
    ("stereoset_binary_bert_predheadonly", False),
    # ("stereoset_binary_bert_all", False),
    # ("stereoset_all_bert_predheadonly", True),
    # ("stereoset_all_bert_all", True)
]

summary = {checkpoint: test_shapley(*checkpoint) for checkpoint in checkpoints}

print(summary)
