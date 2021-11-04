torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:5678" \
  /home/shenli/project/benchmark/script.py
