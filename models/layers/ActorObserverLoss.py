import torch
from torch import nn
from models.layers.EqualizeGradNorm import EqualizeGradNorm
from models.layers.VideoSoftmax import VideoSoftmax
from models.layers.MarginRank import MarginRank
from models.layers.DistRatio import DistRatio
from models.layers.BlockGradient import BlockGradient


VERBOSE = True


def dprint(message, *args):
    if VERBOSE:
        print(message.format(*args))


class ActorObserverLoss(nn.Module):
    def __init__(self, args):
        super(ActorObserverLoss, self).__init__()
        self.loss = globals()[args.subloss]
        self.xstorage = {}
        self.ystorage = {}
        self.zstorage = {}
        self.storage = {}
        self.decay = args.finaldecay
        self.xmax = VideoSoftmax(self.xstorage, args.decay)
        self.ymax = VideoSoftmax(self.ystorage, args.decay)
        self.zmax = VideoSoftmax(self.zstorage, args.decay)
        self.margin = args.margin

    def get_constants(self, ids):
        out = [self.storage[x][0] for x in ids]
        return torch.autograd.Variable(torch.Tensor(out).cuda())

    def update_constants(self, input, weights, ids):
        for x, w, vid in zip(input, weights, ids):
            x, w = x.data[0], w.data[0]
            if vid not in self.storage:
                self.storage[vid] = [x, w]
            else:
                # here J is stored as E[wJ]
                old_x, old_w = self.storage[vid]
                val = (1 - self.decay) * w * x + self.decay * old_w * old_x
                new_weight = (1 - self.decay) * w + self.decay * old_w
                val = val / new_weight
                self.storage[vid] = [val, new_weight]
                if new_weight < 0.0001:
                    print('MILC new_weight is effectively 0')

    def forward(self, dist_a, dist_b, x, y, z, target, ids):
        # Normalize and combine weights
        x = self.xmax(x, ids)
        y = self.ymax(y, ids)
        z = self.zmax(z, ids)
        dist_a, dist_b, x, y, z = EqualizeGradNorm.apply(dist_a, dist_b, x, y, z)
        w = x * y * z

        # update L
        loss = self.loss.apply(dist_a, dist_b, target, self.margin)
        self.update_constants(loss, w, ids)
        k = self.get_constants(ids)
        n = (w.sum() + 0.00001) / w.shape[0]
        final = ((loss - k) * (w / n)).sum()

        dprint('loss before {}', loss.data.sum())
        dprint('loss after {}', (loss.data * w.data / n.data).sum())
        dprint('weight median: {}, var: {}', w.data.median(), w.data.var())

        return final, w.data.cpu()
