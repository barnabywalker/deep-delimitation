import math
import torch

def ntxent(x1, x2, temperature, eps=1e-6):
    x = torch.cat([x1, x2], dim=0)
    x_scores = torch.mm(x, x.t())
    
    sim = torch.exp(x_scores, temperature)
    neg = sim.sum(dim=-1)

    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)

    pos = torch.exp(torch.sum(x1 * x2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    return -torch.log(pos / (neg + eps)).mean()
