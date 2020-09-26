import torch


def margin_loss(outputs, labels, n_classes, dtype=torch.float32):
    margin_positive = torch.Tensor([.9])
    margin_negative = torch.Tensor([.1])
    lambd = torch.Tensor([.5])

    batch_size = len(labels)
    one_hot_labels = torch.zeros(size=(batch_size, n_classes), dtype=dtype)
    for i, label in enumerate(labels):
        label = torch.LongTensor(label)
        one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0,
                                                       index=label, value=1.)

    postive_loss = torch.square(torch.max(torch.zeros_like(outputs),
                                          margin_positive - outputs))
    negative_loss = torch.square(torch.max(torch.zeros_like(outputs),
                                           outputs - margin_negative))

    negative_loss = torch.mul((1. - one_hot_labels), negative_loss)
    postive_loss = torch.mul(one_hot_labels, postive_loss)

    loss = postive_loss + lambd * negative_loss

    return torch.sum(loss) / batch_size
