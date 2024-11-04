import torch
import torch.nn as nn


def cat2ohe(x, a, num_node_types, num_edge_types):
    x = nn.functional.one_hot(x.to(torch.int64), num_classes=num_node_types)
    a = nn.functional.one_hot(a.to(torch.int64), num_classes=num_edge_types)
    return x.to(torch.float), a.to(torch.float)

def ohe2cat(x, a):
    x = torch.argmax(x, dim=-1)
    a = torch.argmax(a, dim=-1)
    return x.to(torch.float), a.to(torch.float)
