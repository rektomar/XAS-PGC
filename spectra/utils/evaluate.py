import torch
import matplotlib.pyplot as plt

from utils.datasets import MIN_E, MAX_E, N_GRID, Identity
# 270, 300, 100

DELTA_E = (MAX_E-MIN_E)/N_GRID


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    rse_tensor = rse_tensor[rse_tensor < rse_tensor.quantile(0.99)]  # filter outliers, needs to be changed to 0.9 when using lognormal standardization

    return torch.mean(rse_tensor)
    
def eval_metrics(model, loaders):

    transform = loaders['transform']

    return {'rse_trn': eval_rse(model, loaders['loader_trn'], transform),
            'rse_val': eval_rse(model, loaders['loader_val'], transform),
            'rse_tst': eval_rse(model, loaders['loader_tst'], transform),
            }

def plot_results(target, prediction, metric, title=''):

    metric_min = torch.min(metric)
    metric_max = torch.quantile(metric, 0.99)

    num_bins = 200
    bin_edges = torch.linspace(metric_min, metric_max, steps=num_bins + 1)

    metric_np = metric.numpy()
    bin_edges_np = bin_edges.numpy()

    perm = torch.argsort(metric)
    energies = torch.linspace(270, 300, 100)

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 4, 1)
    plt.hist(metric_np, bins=bin_edges_np, edgecolor='black', label=f'Average RSE {torch.mean(metric):.3f}')
    plt.title('RSE Histogram')
    plt.xlabel('RSE')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 4, 2)
    id = 0
    plt.plot(energies, prediction[perm][id], label='Reconstructed Spectrum', color='blue')
    plt.plot(energies, target[perm][id], label='Target Spectrum', color='orange')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.title(f'Best, RSE {metric[perm][id]:.3f}')
    plt.legend()


    plt.subplot(1, 4, 3)
    id = torch.argmin(torch.abs(metric[perm] - torch.mean(metric)))
    plt.plot(energies, prediction[perm][id], label='Reconstructed Spectrum', color='blue')
    plt.plot(energies, target[perm][id], label='Target Spectrum', color='orange')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.title(f'Average, RSE {metric[perm][id]:.3f}')
    plt.legend()

    plt.subplot(1, 4, 4)
    id = -1
    plt.plot(energies, prediction[perm][id], label='Reconstructed Spectrum', color='blue')
    plt.plot(energies, target[perm][id], label='Target Spectrum', color='orange')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.title(f'Worst, RSE {metric[perm][id]:.3f}')
    plt.legend()

    plt.suptitle(title, fontsize=16)

    plt.tight_layout()

    plt.savefig('result.png')
    

def eval_visual(model, loader, transform, title):

    rse_tensor = []
    target_tensor = []
    prediction_tensor = []
    for b in loader:
        x = b.float().to(model.device)
        x_hat = model.reconstruct(x)

        x = transform.inverse(x.to('cpu'))
        x_hat = transform.inverse(x_hat.to('cpu'))

        rse_tensor.append(rse(x, x_hat))
        target_tensor.append(x)
        prediction_tensor.append(x_hat)
    

    rse_tensor = torch.cat(rse_tensor)
    mask = rse_tensor < rse_tensor.quantile(0.99)
    rse_tensor = rse_tensor[mask]

    target_tensor = torch.cat(target_tensor)[mask]
    prediction_tensor = torch.cat(prediction_tensor)[mask]

    plot_results(target_tensor, prediction_tensor, rse_tensor, title)


if __name__ == '__main__':
    target = torch.rand((64, 100))
    prediction = torch.rand((64, 100))

    metric = rse(target, prediction)    

    print(metric.shape, metric)