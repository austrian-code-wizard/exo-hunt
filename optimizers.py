import torch.optim as optim

# Returns the SGD optimizer with the given learning rate (and optionally momentum)
def get_sgd_optimizer(model, learning_rate, momentum=0.9):
    return optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=momentum)

# Returns the Adam optimizer with the given learning rate
def get_adam_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

OPTIMIZERS = {
    'sgd': get_sgd_optimizer,
    'adam': get_adam_optimizer
}