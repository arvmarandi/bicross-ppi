import torch
from contextlib import contextmanager

@contextmanager
def torch_manual_seed(seed):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)
