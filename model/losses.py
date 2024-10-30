import numpy as np
import torch


def loss(target, pred, latitude, weights, split_dim=5, level=np.array([925, 850, 700, 500, 200, 100])):
    loss_func = torch.nn.MSELoss(reduction='none')
    loss = loss_func(target, pred).view(target.shape[0], 721, 1440, -1)
    losses = []
    for i in range(target.shape[0]):
        loss[i] *= normalized_latitude_weights(latitude).reshape(-1, 1, 1)
        # TODO: 后续应该是一起算的，但是暂时分开先，graphcast的源码是用xarray，可以分开surface和upper，但是在torch中都
        #  是一样的，所以，只是分开处理就行，然后 再进行 按照level处理upper即可，就是处理后变成变量为 30 + 5， 其中[5，6]过level，然后[5]过surafce即可
        loss_upper = loss[i][:, :, :-split_dim].reshape(loss[i].shape[0], loss[i].shape[1], -1, len(level))
        loss_surface = loss[i][:, :, -split_dim:]
        # upper
        loss_upper *= normalized_level_weights(level, pred.device).reshape(1, 1, 1, -1)
        loss_upper = torch.mean(loss_upper, dim=(0, 1, 3))
        loss_surface = torch.mean(loss_surface, dim=(0, 1))

        losses.append(sum_per_variable_losses(loss_upper, loss_surface, weights))
    return torch.sum(torch.stack(losses, dim=0), dim=0)
    # surface


def sum_per_variable_losses(loss_upper, loss_surface, weights):
    # all_losses = torch.concat([loss_upper, loss_surface], dim=0)
    all_losses = loss_upper
    mse_loss = []
    for i, weight in enumerate(weights):
        mse_loss.append(loss_surface[i] * weight)
    mse_loss = torch.stack(mse_loss)
    return torch.concat([all_losses, mse_loss], dim=0).sum()


def normalized_level_weights(level, device):
    # level = torch.tensor(level, dtype=torch.float32).to(device)
    # return level / level.mean()
    level_sum = level.sum(axis=0)
    return torch.tensor(level, dtype=torch.float32).to(device) / level_sum


def normalized_latitude_weights(latitude):
    # latitude [90,.......,-90] shape: [721,]

    if torch.any(torch.isclose(torch.abs(latitude), torch.tensor(90.0, dtype=latitude.dtype))):
        weights = weight_for_latitude_vector_with_poles(latitude)
    else:
        weights = weight_for_latitude_vector_without_poles(latitude)

    weights = weights / weights.mean()
    return weights


def weight_for_latitude_vector_with_poles(latitude):
    delta_latitude = torch.diff(latitude)[0]
    if (not torch.isclose(torch.max(latitude), torch.tensor(90.)) or
            not torch.isclose(torch.min(latitude), torch.tensor(-90.))):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = torch.cos(torch.deg2rad(latitude)) * torch.sin(torch.deg2rad(delta_latitude / 2))
    # weights[[0, -1]] = torch.sin(torch.deg2rad(torch.tensor(delta_latitude) / 4)) ** 2
    weights[[0, -1]] = torch.sin(torch.deg2rad(delta_latitude.clone().detach()) / 4) ** 2

    return weights


def weight_for_latitude_vector_without_poles(latitude):
    delta_latitude = torch.diff(latitude)[0]
    if (not torch.isclose(torch.max(latitude), torch.tensor(90. - delta_latitude / 2)) or
            not torch.isclose(torch.min(latitude), torch.tensor(-90. + delta_latitude / 2))):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    return torch.cos(torch.deg2rad(latitude))
