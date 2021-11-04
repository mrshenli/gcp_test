import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# see torchrun tutorial: https://pytorch.org/docs/stable/elastic/run.html

def example(rank, world_size):
    print(f"worker{rank} started")
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    print(f"worker{rank} finished")

def main():
    world_size = int(os.getenv("WORLD_SIZE"))
    #local_rank = os.getenv("LOCAL_RANK")
    rank = int(os.getenv("RANK"))
    example(rank, world_size)
    
if __name__=="__main__":
    main()
