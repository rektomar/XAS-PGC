import torch

from utils.datasets import MIN_E, MAX_E, N_GRID, Identity
# 270, 300, 100

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


DELTA_E = (MAX_E-MIN_E)/N_GRID

def rse(target, prediction):
    assert target.shape == prediction.shape

    err = torch.sqrt(torch.sum(torch.square(target - prediction), dim=-1) * DELTA_E)
    norm =  torch.sum(target, dim=-1) * DELTA_E
    return err/norm

def eval_rse(model, loader, transform):
    rse_tensor = []
    for b in loader:
        x = b.float().to(model.device)
        x_hat = model.reconstruct(x)

        x = transform.inverse(x.to('cpu'))
        x_hat = transform.inverse(x_hat.to('cpu'))
        rse_tensor.append(rse(x, x_hat))

    rse_tensor = torch.cat(rse_tensor)
    rse_tensor = rse_tensor[rse_tensor < rse_tensor.quantile(0.99)]  # filter outliers

    return torch.mean(rse_tensor)
    
def eval_metrics(model, loaders):

    transform = loaders['transform']

    return {'rse_trn': eval_rse(model, loaders['loader_trn'], transform),
            'rse_val': eval_rse(model, loaders['loader_val'], transform),
            'rse_tst': eval_rse(model, loaders['loader_tst'], transform),
            }

if __name__ == '__main__':
    target = torch.rand((64, 100))
    prediction = torch.rand((64, 100))

    metric = rse(target, prediction)    

    print(metric.shape, metric)