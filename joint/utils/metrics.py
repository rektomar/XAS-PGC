import torch


# 270, 300, 100

deltaE = (300-270)/100

def rse(target, prediction):
    assert target.shape == prediction.shape

    err = torch.sqrt(torch.sum((target - prediction)**2, dim=-1) * deltaE)
    norm =  torch.sum(target, dim=-1) * deltaE
    return err/norm


if __name__ == '__main__':
    target = torch.rand((64, 100))
    prediction = torch.rand((64, 100))

    metric = rse(target, prediction)    

    print(metric.shape, metric)
    