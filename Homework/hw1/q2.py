###Q2: Parameter Server###
###please implement PS method, using  parameter update&optimizor states all be held in rank0###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def server(params, opt, world):
    # ---- aggregate grads from workers ----
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous() #usage: tranfer a list tensor to one 1-D tensor
    ##here, you should generate one big 1-D tensor containing all parameters to make the transfer process easy
    agg = flat_grad.clone() #agg as a aggregated counter to record sum gradients

    #                                                                   #
    #                                                                   #
    # your code here: receive gradients form worker, and add them to agg#
    #                                                                   #
    #                                                                   #
    
    for src in range(1, world):
        recv_buf = torch.empty_like(flat_grad)
        dist.recv(recv_buf, src=src)
        agg.add_(recv_buf)
    agg.div_(world) #average the gradients


    synced_grads = _unflatten_dense_tensors(agg, [p.grad for p in params])
    # ---- set averaged grads locally & step ----
    for g, s in zip([p.grad for p in params], synced_grads):
        g.copy_(s)
    opt.step()

    # ---- broadcast updated params for this subset ----
    flat_param = _flatten_dense_tensors([p.data for p in params]).contiguous()
    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D parameter tensor to all workers   #
    #                                                                   #
    #                                                                   #
    for dst in range(1, world):
        dist.send(flat_param, dst=dst)

def worker(params):
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous()
    # ---- push grads to server ----

    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D gradient to server
    #                                                                   #
    #                                                                   #
    dist.send(flat_grad, dst=0)
    # ---- receive updated params, write into local model ----
    
    #                                                                   #
    #                                                                   #
    # your code here: please get correct 1-D packed parameter from server
    #           And then unpacked it and store in synced_params
    #                                                                   #
    synced_params = None #you should  assign correct value for synced_params#
    recv_buf = torch.empty_like(flat_grad)
    dist.recv(recv_buf, src=0)
    synced_params = _unflatten_dense_tensors(recv_buf, [p.data for p in params])

    # ---- syncronize the parameters ----
    for p, s in zip(params, synced_params):
        p.data.copy_(s)

def PS_grads_(model,world_size=None, rankid=None, opt=None):
    """
    Synchronous PS step:
      - Rank 0: receive grads, sum/avg, set grads, opt.step(), broadcast updated params.
      - Rank >0: send grads, receive updated params, write into local model.
    Only processes the subset of parameters with non-None grads.
    """
    world = dist.get_world_size() if world_size is None else world_size
    rank  = dist.get_rank() if rankid is None else rankid
    
    # Fast path: single process
    if world == 1:
        opt.step()
        return

    #not necessary, as in most case, params won't be empty list...
    # Collect params that participated in this backward pass
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        # No grads this step; only server might still want to advance schedulers, etc.
        if rank == 0:
            opt.step()
        return

    if rank == 0:
        server(params, opt, world)

    else:
        worker(params)

    # Optional hard step boundary
    dist.barrier()