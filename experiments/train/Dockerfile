FROM ghcr.io/henryscheible/nlpcore:v1.0.0
ARG TOKEN
ARG MODEL
ARG DATASET
ARG GPU_CARD
ARG FINETUNE
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
COPY ./train.py /workspace
ENV NVIDIA_VISIBLE_DEVICES=$GPU_CARD
ENV CUDA_VISIBLE_DEVICES=$GPU_CARD
ENV MODEL=$MODEL
ENV DATASET=$DATASET
ENV FINETUNE=$FINETUNE
CMD ["python3", "/workspace/train.py"]
