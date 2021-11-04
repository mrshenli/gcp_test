/home/shenli/miniconda3/bin/torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="instance-shen-debugging-2:5678" \
  /home/shenli/project/benchmark/script.py
