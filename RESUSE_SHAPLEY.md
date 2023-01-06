# How to Reuse Shapley Probing Code for Other Tasks

* The training script for the shapley probing experiment can be found in `/experiments/shapley/train.py`, but it's probably easier to start from scratch using the shapley tools provided in `nplcore`. Here are instructions for how to do that.
* The function `get_shapley` in https://github.com/henryscheible/nlpcore/blob/main/nlpcore/shapley.py provides an implementation of the shapley algorithm. The only part of the function that will need to be changed is the AutoModel class. Currently it is AutoModelForSequenceClassification but it will be changed for a new task. 
* After that, `get_shapley` can be called with the following arguments:
  * `eval_dataloader`: torch `DataLoader` object for the evaluation data (already tokenized)
  * `chechpoint`: Model checkpoint to download from HuggingFace hub
  * `num_samples`: Number of samples. We have been using 250