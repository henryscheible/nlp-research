FROM ghcr.io/henryscheible/nlpcore:v1.1.1
ARG TOKEN
ARG MODEL
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
COPY ./train.py /workspace
ENV NVIDIA_VISIBLE_DEVICES=1
ENV CUDA_VISIBLE_DEVICES=1
ENV MODEL=$MODEL
CMD ["python3", "/workspace/train.py"]
