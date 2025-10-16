###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #               
    #                                                     #
    for i in range(world - 1):
        send_req = dist.isend(tensor=chunks[(rank - i) % world], dst=right)
        r = dist.irecv(tensor=tmp, src=left)
        send_req.wait()
        r.wait()
        chunks[(rank - i - 1) % world].add_(tmp)
        
    return
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    idx = current
    for _ in range(world - 1):
        send_req = dist.isend(tensor=chunks[idx], dst=right)
        rec_ = dist.irecv(tensor=tmp, src=left)
        send_req.wait()
        rec_.wait()
        idx = (idx - 1 + world) % world
        chunks[idx].copy_(tmp)
    return

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = dist.get_world_size() if world_size is None else world_size
    if world == 1: return tensor
    rank = dist.get_rank() if rankid is None else rankid

    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padded_flat = None # modify this line and fill correct value into padded_flat
    total_size = chunk * world
    if n < total_size:
        padded_flat = torch.zeros(total_size, dtype=flat.dtype, device=flat.device)
        padded_flat[:n].copy_(flat)
    else:
        padded_flat = flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    tmp = torch.empty_like(chunks[0])
    reduce_scatter(chunks, tmp, world, rank, left, right)
    dist.barrier()
    current = (rank - (world - 1)) % world
    all_gather(chunks, tmp, current, world, rank, left, right)
    dist.barrier()
    if padded_flat is not flat:
        flat.copy_(padded_flat[:n])
    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return