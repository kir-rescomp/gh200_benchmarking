import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from datetime import timedelta
import os
transformers.utils.logging.set_verbosity_error()

MODEL_ID = "bert-base-uncased"
CACHE_DIR = "/gpfs3/well/kir/projects/mirror/LLM/huggingface/google-bert-berse-case-uncased" # Use pre-downloaded models
BACKEND = 'nccl' # change to 'nccl' to use NCCL backend
BATCH_SIZE = 32
NUM_SAMPLES = 64  # Total number of samples to process
TRAINING_STEPS = NUM_SAMPLES // BATCH_SIZE
DEVICE = f"cuda:{os.environ['LOCAL_RANK']}"

def init_process(backend=BACKEND):
    print(f"Initializing distributed training rank {os.environ.get('RANK')} with backend: {backend} on device: {DEVICE}")
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=60*5), world_size=int(os.environ['WORLD_SIZE']))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    world_size = dist.get_world_size()
    if dist.get_rank() == 0:
        print(f"Distributed training initialized with {world_size} processes using backend {backend}.")

def benchmark():
    model_path = CACHE_DIR
    tokenizer = BertTokenizer.from_pretrained(model_path)
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    per_gpu_batch_size = BATCH_SIZE // world_size

    if dist.get_rank() == 0:
        print(f"Running benchmark with world size: {world_size}, batch size: {BATCH_SIZE}, per GPU batch size: {per_gpu_batch_size}, training steps: {TRAINING_STEPS}")

    model = BertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters())

    # Separate data per worker
    start_idx = local_rank * per_gpu_batch_size
    end_idx = start_idx + per_gpu_batch_size
    # Create synthetic training data
    texts = [f"This is sample sentence {i} for benchmarking BERT." for i in range(start_idx, end_idx)]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, use_fast_tokenizer=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    labels = torch.ones(per_gpu_batch_size, dtype=torch.long).to(DEVICE)

    dist.barrier()
    start_time = time.time()

    for _ in range(TRAINING_STEPS):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels) # forward pass
        loss = outputs.loss
        loss.backward() # backward pass
        optimizer.step()

    dist.barrier()
    end_time = time.time()

    if dist.get_rank() == 0:
        print(f"Time taken for {TRAINING_STEPS} forward and backward pass(es) with BATCH_SIZE={BATCH_SIZE} on {world_size} workers: {end_time - start_time} seconds")

if __name__ == "__main__":
    init_process()
    benchmark()
    dist.destroy_process_group()
