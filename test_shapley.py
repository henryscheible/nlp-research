import json

import evaluate
import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import HfApi, hf_hub_download
from nlpcore.bias_datasets.stereoset import load_processed_stereoset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt


def pull_contribs(checkpoint):
    res = requests.get(f"https://huggingface.co/henryscheible/{checkpoint}/raw/main/contribs.txt")
    return json.loads(res.text)


def get_positive_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution > 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu")


def get_negative_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution < 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu")


def get_bottom_up_masks(contribs):
    contribs = np.random.rand(144)
    sorted_indices = np.argsort(contribs)
    tmp = (np.asarray(sorted_indices) < 0).sum()
    masks = [np.zeros(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 1
        masks += [new_mask]
    return tmp, [torch.tensor(mask).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def get_top_down_masks(contribs):
    contribs = np.random.rand(144)
    sorted_indices = np.argsort(contribs)
    masks = [np.ones(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 0
        masks += [new_mask]
    return [torch.tensor(mask).reshape(12, 12).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def evaluate_model(eval_loader, model, mask=None):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    metric = evaluate.load('accuracy')

    for eval_batch in eval_loader:
        eval_batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in eval_batch.items()}
        with torch.no_grad():
            outputs = model(**eval_batch, head_mask=mask) if mask is not None else model(**eval_batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=eval_batch["labels"])

    return metric.compute()["accuracy"]


def test_shapley(checkpoint, include_unrelated):
    REPO = "henryscheible/" + checkpoint
    print(f"=======CHECKPOINT: {checkpoint}==========")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # _, eval_loader = load_processed_stereoset(tokenizer, include_unrelated=include_unrelated)
    # model = AutoModelForSequenceClassification.from_pretrained(REPO)
    # base_acc = evaluate_model(eval_loader, model)
    # progress_bar = tqdm(range(144))

    # contribs = pull_contribs(checkpoint)

    bottom_up_results = []
    # vline, bottom_up_contribs = get_bottom_up_masks(contribs)
    # for mask in bottom_up_contribs:
    #     bottom_up_results += [evaluate_model(eval_loader, model, mask=mask)]
    #     progress_bar.update(1)

    progress_bar = tqdm(range(144))

    # top_down_results = []
    # for mask in get_top_down_masks(contribs):
    #     top_down_results += [evaluate_model(eval_loader, model, mask=mask)]
    #     progress_bar.update(1)

    return {
        "base_acc": base_acc,
        "bottom_up_results": bottom_up_results,
        "top_down_results": top_down_results,
        "vline": vline
    }


def generate_plots(checkpoint, results):
    # Bottom Up Plot
    processed_data = pd.DataFrame({
        'heads': np.arange(145),
        'bottom_up': results["bottom_up_results"],
        'top_down': results["top_down_results"]
    })
    plt.figure(dpi=300)
    bottom_up_plot = sns.lineplot(x='heads', y='value', hue='variable',
                                  data=pd.melt(processed_data, ['heads']))
    bottom_up_plot.set(
        title=f"{checkpoint}",
        xlabel="# of Attention Heads Added",
        ylabel="Accuracy"
    )
    plt.axvline(results["vline"], 0, 1)
    plt.savefig(f'{checkpoint}-accuracy-test.png')

    plt.figure(dpi=300)
    heatmap = sns.heatmap(np.array(pull_contribs(checkpoint)).reshape((12, 12)), cmap="GiR")
    heatmap.set(
        title=f"{checkpoint} Attention Head Contributions"
    )
    plt.savefig(f'{checkpoint}-heatmap.png')
    api = HfApi()
    api.upload_file(
        path_or_fileobj=f'{checkpoint}-accuracy-test.png',
        path_in_repo=f'{checkpoint}-accuracy-test.png',
        repo_id=f"henryscheible/{checkpoint}",
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=f'{checkpoint}-heatmap.png',
        path_in_repo=f'{checkpoint}-heatmap.png',
        repo_id=f"henryscheible/{checkpoint}",
        repo_type="model",
    )


checkpoints = [
    ("stereoset_binary_bert_classifieronly", False),
    ("stereoset_binary_bert_finetuned", False),
    ("stereoset_all_bert_classifieronly", True),
    ("stereoset_all_bert_finetuned", True)
]

for checkpoint in checkpoints:
    print(f"===========CHECKPOINT: {checkpoint}=============")
    # results = test_shapley(*checkpoint)

    api = HfApi()
    results = {"base_acc": 0.7182103610675039, "bottom_up_results": [0.5054945054945055, 0.5054945054945055, 0.5054945054945055, 0.5054945054945055, 0.5047095761381476, 0.5054945054945055, 0.5054945054945055, 0.5062794348508635, 0.5054945054945055, 0.5054945054945055, 0.5054945054945055, 0.5062794348508635, 0.5023547880690737, 0.5039246467817896, 0.5062794348508635, 0.5141287284144427, 0.5102040816326531, 0.5125588697017268, 0.5086342229199372, 0.5086342229199372, 0.5235478806907379, 0.5243328100470958, 0.5345368916797488, 0.533751962323391, 0.5392464678178964, 0.5455259026687598, 0.5463108320251178, 0.5384615384615384, 0.5392464678178964, 0.542386185243328, 0.543171114599686, 0.5486656200941915, 0.5588697017268446, 0.554160125588697, 0.5565149136577708, 0.5612244897959183, 0.5557299843014128, 0.5620094191522763, 0.554160125588697, 0.597331240188383, 0.5989010989010989, 0.5941915227629513, 0.6083202511773941, 0.6059654631083202, 0.6036106750392465, 0.6059654631083202, 0.5996860282574569, 0.6004709576138147, 0.6020408163265306, 0.598116169544741, 0.609105180533752, 0.6051805337519623, 0.6145996860282574, 0.6020408163265306, 0.6036106750392465, 0.6059654631083202, 0.597331240188383, 0.6004709576138147, 0.5934065934065934, 0.6059654631083202, 0.6004709576138147, 0.6098901098901099, 0.6083202511773941, 0.6083202511773941, 0.6138147566718996, 0.6208791208791209, 0.6138147566718996, 0.5965463108320251, 0.6012558869701727, 0.5910518053375197, 0.6004709576138147, 0.5926216640502355, 0.5871271585557299, 0.5941915227629513, 0.5934065934065934, 0.5957613814756672, 0.5965463108320251, 0.598116169544741, 0.6004709576138147, 0.5989010989010989, 0.5934065934065934, 0.6036106750392465, 0.6020408163265306, 0.598116169544741, 0.5949764521193093, 0.6051805337519623, 0.6020408163265306, 0.6004709576138147, 0.6028257456828885, 0.5996860282574569, 0.6075353218210361, 0.6067503924646782, 0.619309262166405, 0.619309262166405, 0.6224489795918368, 0.6208791208791209, 0.6216640502354788, 0.6295133437990581, 0.6334379905808477, 0.6334379905808477, 0.630298273155416, 0.6295133437990581, 0.6357927786499215, 0.6718995290423861, 0.673469387755102, 0.6671899529042387, 0.6679748822605965, 0.6616954474097331, 0.6632653061224489, 0.6671899529042387, 0.6601255886970173, 0.6593406593406593, 0.6640502354788069, 0.6695447409733124, 0.673469387755102, 0.6789638932496075, 0.6836734693877551, 0.6789638932496075, 0.6781789638932496, 0.6797488226059655, 0.6868131868131868, 0.6836734693877551, 0.6844583987441131, 0.6758241758241759, 0.6789638932496075, 0.6797488226059655, 0.6821036106750392, 0.6844583987441131, 0.6813186813186813, 0.6797488226059655, 0.6750392464678179, 0.6711145996860283, 0.673469387755102, 0.6726844583987441, 0.673469387755102, 0.6711145996860283, 0.6758241758241759, 0.6836734693877551, 0.6875981161695447, 0.6923076923076923, 0.7032967032967034, 0.707221350078493, 0.7158555729984302, 0.7127158555729984, 0.7182103610675039], "top_down_results": [0.7182103610675039, 0.717425431711146, 0.716640502354788, 0.7244897959183674, 0.7237048665620094, 0.7252747252747253, 0.7237048665620094, 0.7252747252747253, 0.728414442700157, 0.7291993720565149, 0.7276295133437991, 0.7260596546310832, 0.7260596546310832, 0.7276295133437991, 0.728414442700157, 0.7244897959183674, 0.7244897959183674, 0.7268445839874411, 0.7252747252747253, 0.7260596546310832, 0.7268445839874411, 0.7213500784929356, 0.7229199372056515, 0.7229199372056515, 0.7252747252747253, 0.7221350078492935, 0.7260596546310832, 0.728414442700157, 0.7260596546310832, 0.7252747252747253, 0.7252747252747253, 0.7252747252747253, 0.7244897959183674, 0.7244897959183674, 0.7276295133437991, 0.7252747252747253, 0.7260596546310832, 0.7213500784929356, 0.7237048665620094, 0.7150706436420722, 0.7182103610675039, 0.7158555729984302, 0.7142857142857143, 0.7095761381475667, 0.7087912087912088, 0.7087912087912088, 0.7040816326530612, 0.7087912087912088, 0.7009419152276295, 0.7025117739403454, 0.7040816326530612, 0.7032967032967034, 0.706436420722135, 0.7080062794348508, 0.7087912087912088, 0.7095761381475667, 0.7048665620094191, 0.706436420722135, 0.7009419152276295, 0.7048665620094191, 0.707221350078493, 0.7056514913657771, 0.7040816326530612, 0.7040816326530612, 0.7056514913657771, 0.7080062794348508, 0.7119309262166404, 0.7095761381475667, 0.7040816326530612, 0.7017268445839875, 0.7017268445839875, 0.7009419152276295, 0.6970172684458399, 0.6938775510204082, 0.6891679748822606, 0.6907378335949764, 0.6938775510204082, 0.6923076923076923, 0.6860282574568289, 0.6844583987441131, 0.6899529042386185, 0.6875981161695447, 0.6821036106750392, 0.6836734693877551, 0.6828885400313972, 0.6875981161695447, 0.6805337519623234, 0.6805337519623234, 0.6758241758241759, 0.6687598116169545, 0.6656200941915228, 0.6671899529042387, 0.6703296703296703, 0.6679748822605965, 0.6679748822605965, 0.6562009419152276, 0.6491365777080063, 0.6459968602825745, 0.6514913657770801, 0.652276295133438, 0.6507064364207221, 0.6436420722135008, 0.6444270015698587, 0.6420722135007849, 0.6326530612244898, 0.6326530612244898, 0.6459968602825745, 0.6499215070643642, 0.6507064364207221, 0.6499215070643642, 0.6569858712715856, 0.6546310832025117, 0.6491365777080063, 0.6373626373626373, 0.6514913657770801, 0.6483516483516484, 0.6475667189952904, 0.6554160125588697, 0.5886970172684458, 0.5800627943485086, 0.5745682888540031, 0.5777080062794349, 0.5784929356357927, 0.5784929356357927, 0.5722135007849294, 0.5612244897959183, 0.5949764521193093, 0.5690737833594977, 0.5486656200941915, 0.5416012558869702, 0.5518053375196232, 0.554945054945055, 0.521978021978022, 0.5156985871271585, 0.5204081632653061, 0.5125588697017268, 0.5070643642072213, 0.5062794348508635, 0.5047095761381476, 0.5054945054945055, 0.5054945054945055, 0.5054945054945055, 0.5054945054945055, 0.5054945054945055, 0.5054945054945055]}
    generate_plots(checkpoint[0], results)
