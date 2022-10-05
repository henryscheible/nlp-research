#!/usr/bin/env sh
conda_env="nlp_env"
ref="main"
name="$1"
host="henry@dsail2.cs.dartmouth.edu"
now="$(date +"%m-%d-%Y-%T")"
while getopts e:r:n:h: flag
do
    case "${flag}" in
        e) conda_env=${OPTARG};;
        r) ref=${OPTARG};;
        n) name=${OPTARG};;
        h) host=${OPTARG};;
        *) echo "Invalid flag passed";;
    esac
done
ssh "$host" "mkdir ~/experiments/$1_$now"
scp ./run_experiment.sh "$host:~/experiments/$1_$now/"
scp "experiments/$1/train.py" "$host:~/experiments/$1_$now/"
ssh "$host" "~/experiments/$1_$now/run_experiment.sh ."


