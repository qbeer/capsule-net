import torch as T
import torch.nn.functional as F
from torch.nn import Conv2d


class CapsuleNetwork(T.nn.Module):
    def __init__(self, batch_size, capsule_dims=[8, 16], n_caps=[1152, 10]):
        super(CapsuleNetwork, self).__init__()
        self.capsule_dims = capsule_dims
        self.n_caps = n_caps
        self.conv1 = Conv2d(in_channels=1,
                            out_channels=256,
                            kernel_size=9,
                            stride=1)
        self.conv2 = Conv2d(in_channels=256,
                            out_channels=256,
                            kernel_size=9,
                            stride=2)

        stddev = .01
        self.transformation_weights = T.nn.Parameter(T.normal(
            mean=0,
            std=stddev,
            size=(1, n_caps[0], n_caps[1], capsule_dims[1], capsule_dims[0])),
            requires_grad=True)

        self.raw_weights = T.nn.Parameter(
            T.zeros(size=(batch_size, self.n_caps[0], self.n_caps[1], 1, 1)))

    def _squash(self, tensor, axis=-1, epsilon=1e-8):
        norm = self._norm(tensor, axis=axis, keepdims=True)
        squash = T.square(norm) / (1. + T.square(norm))
        unit = tensor / norm
        return unit * squash

    def _norm(self, tensor, axis=-1, keepdims=False, epsilon=1e-8):
        squared_norm = T.sum(T.square(tensor), axis=axis, keepdims=keepdims)
        return T.sqrt(squared_norm + epsilon)

    def forward(self, inputs):
        x, y = inputs

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        batch_size, channels, width, heigth = x.shape
        x = T.reshape(x, (batch_size, width * heigth * channels //
                          self.capsule_dims[0], self.capsule_dims[0]))
        x = self._squash(x)

        x = x.view(batch_size, self.n_caps[0], 1, self.capsule_dims[0],
                   1).repeat(1, 1, self.n_caps[1], 1, 1)

        tiled_transformation_weights = self.transformation_weights.repeat(
            batch_size, 1, 1, 1, 1)

        routing_weights = F.softmax(self.raw_weights, dim=2)

        caps2 = T.matmul(tiled_transformation_weights, x)
        weighted_preds = T.mul(routing_weights, caps2)
        weighted_sum = T.sum(weighted_preds, axis=1, keepdims=True)
        caps2_round1 = self._squash(weighted_sum, axis=-2)

        caps2_round1_tiled = caps2_round1.repeat(1, self.n_caps[0], 1, 1, 1)

        agreement = T.matmul(caps2.transpose(3, 4), caps2_round1_tiled)
        raw_weights_2 = T.add(self.raw_weights, agreement)
        routing_weights = F.softmax(raw_weights_2, dim=2)

        weighted_preds = T.mul(routing_weights, caps2)
        weighted_sum = T.sum(weighted_preds, axis=1, keepdims=True)
        caps2_round2 = self._squash(weighted_sum, axis=-2)

        caps2_round2_tiled = caps2_round2.repeat(1, self.n_caps[0], 1, 1, 1)

        agreement = T.matmul(caps2.transpose(3, 4), caps2_round2_tiled)
        raw_weights_3 = T.add(self.raw_weights, agreement)
        routing_weights = F.softmax(raw_weights_3, dim=2)

        weighted_preds = T.mul(routing_weights, caps2)
        weighted_sum = T.sum(weighted_preds, axis=1, keepdims=True)
        caps2_round3 = self._squash(weighted_sum, axis=-2)

        caps2 = caps2_round3.squeeze()

        caps2_normed = self._norm(caps2)
        preds = T.argmax(caps2_normed, axis=-1).squeeze()

        return preds, caps2, caps2_normed
