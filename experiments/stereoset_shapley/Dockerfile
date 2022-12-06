FROM ghcr.io/henryscheible/nlpcore:v1.0.0
ARG TOKEN
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
COPY ./train.py /workspace
ARG FINETUNE
ARG NUM_SAMPLES
ENV FINETUNE=$FINETUNE
ENV NUM_SAMPLES=$NUM_SAMPLES
ENV NVIDIA_VISIBLE_DEVICES=1
ENV CUDA_VISIBLE_DEVICES=1
CMD ["python3", "/workspace/train.py"]
