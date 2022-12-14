# Instructions for Running Shapley Probing

1. Clone this repository
2. Create a huggingface command line token and add it to the environment with ```export HF_TOKEN="<YOUR TOKEN>"```
3. Edit the docker contexts variables at the top of `groups/shapley.json` to point to docker daemons that you have access to. (this probably amounts to changing the user from "henry" if you are still running on DSAIL). For this to work, you must have set an SSH key to log into the server instead of authenticating with a password.
4. Inside the experiments list in `groups/shapley.json`, the following attributes are given for each experiment:
    * `name` specifies the name of the container to be created
    * `image` specifies the docker image to be built. The Dockerfile is searched for at `experiments/<image name>/Dockerfile`
    * `context` the docker context to run the container on. Must be one of the contexts specified in the context list at the top of the file
    * `card` The GPU card to run the experiment on
    * `buildargs` build arguments that are passed to the docker image. For shapley probing, these are `CHECKPOINT`, `DATASET`, and `NUM_SAMPLES`
5. Launch the experiment group by running `python3 run_experiments.py launch groups/shapley.json`.
6. View docker logs from each experiment in the group by running `python3 run_experiments.py monitor groups/shapley.json`.
7. Stop and cleanup all experiments in the group by running `python3 run_experiments.py stop groups/shapley.json`. (if weird errors occur when attempting to launch the group subsequent times, try stopping it first)