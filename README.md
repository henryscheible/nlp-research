# NLP Research

This repository consists of an assortment of natural language processing experiments, mostly about research into bias in pretrained models such as BERT. 
It also contains a containerization system for quickly starting and monitoring multiple experiments at the same time as well as ensuring experiment reproducibility

## Experiments
Each experiment is a directory in `experiments` consisting of a Dockerfile, possibly with additional python scripts. To run the experiment, the docker image is built. This setup is extremely versatile because an experiment can extend another experiment by using the parent experiment as the starting point for building the docker image. Thus, some experiments images are extremely lightweight, merely modifying a few environment variables from the parent image, while some contain significant scripts

All experiments are based on the `nlpcore-local` image which is built from the docker environment in [this](https://github.com/henryscheible/nlpcore) repository. This image installs essential packages like PyTorch and HuggingFace and also installs the `nlpcore` package which contains code that is shared between a significant number of experiments. By extracting common code to this package, `train.py` files in experiments can be made extremely short and clear, allowing for easy edits and less bugs.

## Visualizations
The visualizations directory is currently a work in progress, but contains graphs and images from experiment results in Jupyter Notebooks. 
By creating visualizations in notebooks instead of inside containerized experiments, debugging is made significantly easier. This is possible because visualization code does not have complicated dependencies.

## Runtime Scripts
### Iteration 1: Shell Script
`run_experiment.sh` selects the correct docker context (can be a remote machine), builds the docker images specified in `experiments.txt`, and starts containers for each image. This is useful for running a few experiments at a time and for simple experiments, but for running a group of experiments on different machines and/or gpus, the python script is more useful

### Iteration 2: Python Script
`run_experiments.py` iterates through a provided JSON file and runs each experiment inside. This file specifies the docker context (essentiall the machine), the gpu card, and the build arguments separately for each experiment, allowing for complicated experiment groups to be defined and launched easily.
This script also has a 'monitor' mode, which loops through already created containers and returns the most recent logged line from each, significantly reducing the number of docker commands necessary to retrieve that information.