#!/usr/bin/env sh
conda_env="nlp_env"
ref="main"
name="$1"
now="$(date +"%m-%d-%Y-%T")"
while getopts e:r: flag
do
    case "${flag}" in
        e) conda_env=${OPTARG};;
        r) ref=${OPTARG};;
        n) name=${OPTARG};;
        *) echo "Invalid flag passed";;
    esac
done
conda env activate "$conda_env"
conda install python
python -m pip uninstall -y nlpcore
python -m pip install "nlpcore @ https://github.com/henryscheible/nlpcore/archive/$ref.zip"
screen -S -d -m "$name" python train.py
