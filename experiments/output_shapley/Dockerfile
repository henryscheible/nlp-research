FROM nlpcore:latest
ARG TOKEN
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
COPY ./train.py /workspace
ARG CHECKPOINT
ARG DATASET
ARG NUM_SAMPLES
ENV CHECKPOINT=$CHECKPOINT
ENV DATASET=$DATASET
ENV NUM_SAMPLES=$NUM_SAMPLES
CMD ["python3", "/workspace/train.py"]
