import torch


def margin_loss(outputs, one_hot_labels):
    margin_positive = .9
    margin_negative = .1
    lambd = .5

    batch_size = len(one_hot_labels)

    postive_loss = torch.square(
        torch.max(torch.zeros_like(outputs), margin_positive - outputs))
    negative_loss = torch.square(
        torch.max(torch.zeros_like(outputs), outputs - margin_negative))

    negative_loss = torch.mul((1. - one_hot_labels), negative_loss)
    postive_loss = torch.mul(one_hot_labels, postive_loss)

    loss = postive_loss + lambd * negative_loss

    return torch.sum(loss) / batch_size
