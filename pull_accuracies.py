import requests
import sys
import json

for launch_file in sys.argv[1:]:
    with open(launch_file, "r") as file:
        data = json.loads("".join(file.readlines()))

    for experiment in data["experiments"]:
        validation = json.loads(requests.get(f"https://huggingface.co/henryscheible/{experiment['name']}/raw/main/validation.json").text)
        print(f"{experiment['name']} \t| {validation['best_model_acc']}")

