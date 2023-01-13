FROM ghcr.io/henryscheible/nlpcore:v2.0.0
ARG TOKEN
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
RUN \
  apt-get update && \
  apt-get install -y git
RUN apt-get install git-lfs
ARG GPU_CARD
ARG TASK
ARG EPOCHS
COPY ./train.py /workspace
ENV NVIDIA_VISIBLE_DEVICES=$GPU_CARD
ENV CUDA_VISIBLE_DEVICES=$GPU_CARD
ENV TOKEN=$TOKEN
ENV TASK=$TASK
ENV EPOCHS=$EPOCHS
CMD python3 /workspace/train.py --model_name_or_path bert-base-uncased --task_name $TASK --do_eval true --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs $EPOCHS --output_dir /tmp/$TASK/ --hub_model_id eval_v4_$TASK --push_to_hub true
