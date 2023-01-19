
# distributed training
torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  --nproc_per_node=4 \
  project/validate.py
  