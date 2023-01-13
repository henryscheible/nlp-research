import requests
import sys
import json

names_to_metrics = {
    "cola": ["eval_matthews_correlation"],
    "sst2": ["eval_accuracy"],
    "mrpc": ["eval_f1", "eval_accuracy"],
    "stsb": ["eval_pearson", "eval_spearmanr"],
    "qqp": ["eval_accuracy", "eval_f1"],
    "mnli": ["eval_accuracy_matched", "eval_accuracy_mismatched"],
    "qnli": ["eval_accuracy"],
    "rte": ["eval_accuracy"],
    "wnli": ["eval_accuracy"]
}

names_to_masked_names = {}

print(f"{'Task Name':<15} | {'Metric(s)':<50} | {'Base':<12} | {'Masked':<12} |")
print(f"{''.join(['-'] * 16)}|{''.join(['-'] * 52)}|{''.join(['-'] * 14)}|{''.join(['-'] * 14)}|")
for name, metrics in names_to_metrics.items():
    printstr = f"{name:<15} | {'/'.join(metrics):<50} |"
    try :
        validation = json.loads(requests.get(f"https://huggingface.co/henryscheible/{name}/raw/main/eval_results.json").text)
        printstr += f" {'/'.join([str(round(100 * validation[metric], 2)) for metric in metrics]):<12} |"
    except json.decoder.JSONDecodeError:
        printstr += f" {'Not Found':<12} |"
    try :
        validation = json.loads(requests.get(f"https://huggingface.co/henryscheible/eval_masked_v4_{name}/raw/main/eval_results.json").text)
        printstr += f" {'/'.join([str(round(100 * validation[metric], 2)) for metric in metrics]):<12} |"
    except json.decoder.JSONDecodeError:
        printstr += f" {'Not Found':<12} |"
    print(printstr)


