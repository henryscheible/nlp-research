FROM ghcr.io/henryscheible/nlpcore:v2.0.0
RUN python3 -m pip install sklearn
RUN \
  apt-get update && \
  apt-get install -y git
RUN apt-get install git-lfs
ARG TOKEN
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
COPY ./train.py /workspace
ARG MODEL
ARG DATASET
ARG GPU_CARD
ARG FINETUNE
ENV NVIDIA_VISIBLE_DEVICES=$GPU_CARD
ENV CUDA_VISIBLE_DEVICES=$GPU_CARD
ENV MODEL=$MODEL
ENV DATASET=$DATASET
ENV FINETUNE=$FINETUNE
ENV EPOCHS=5
CMD ["python3", "/workspace/train.py"]
