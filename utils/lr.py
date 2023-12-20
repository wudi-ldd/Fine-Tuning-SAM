from torch.optim.lr_scheduler import CosineAnnealingLR

def create_cosine_annealing_scheduler(optimizer, T_max, eta_min):
    """
    Create a cosine annealing scheduler.

    :param optimizer: The optimizer.
    :param T_max: The scheduler's period (e.g., the total number of training epochs).
    :param eta_min: The minimum value of the learning rate.
    :return: A configured cosine annealing scheduler.
    """
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


