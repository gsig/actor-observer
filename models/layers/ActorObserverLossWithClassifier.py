import torch
import torch.nn as nn
from models.layers.ActorObserverLoss import ActorObserverLoss
from models.utils import dprint


def var_subset(var, inds):
    inds = torch.autograd.Variable(torch.LongTensor(inds).cuda())
    out = [x.index_select(0, inds) for x in var]
    return out


def hack(inds1, inds2):
    # Fix for avoiding degenerate autograd graph building
    # that causes out of gpu memory
    fix0 = 1.0
    fixweights = 1.0
    if len(inds1) == 0:
        inds1.append(0)
        fix0 = 0.0
        fixweights = 0.0
    if len(inds2) == 0:
        inds2.append(0)
        fix0 = 0.0

    def fix(final0, weights):
        return final0 * fix0, weights * fixweights

    return fix


class ActorObserverLossWithClassifier(ActorObserverLoss):
    def __init__(self, args):
        super(ActorObserverLossWithClassifier, self).__init__(args)
        self.clsloss = nn.CrossEntropyLoss(reduce=False).cuda()
        self.clsweight = args.clsweight

    def forward(self, dist_a, dist_b, x, y, z, cls, target, ids):
        if not cls.volatile:
            cls.register_hook(lambda grad: dprint('cls norm: {}', grad.data.norm()))

        inds1 = [i for i, t in enumerate(target) if t.data[0] > 0]
        inds2 = [i for i, t in enumerate(target) if not t.data[0] > 0]
        dprint('#triplets: {} \t #class: {}', len(inds1), len(inds2))
        fix = hack(inds1, inds2)

        # ActorObserverLoss
        final, weights = [], torch.zeros(x.shape[0],)
        if len(inds1) > 0:
            vars1 = var_subset([dist_a, dist_b, x, y, z, target], inds1)
            vars1 += [[ids[i] for i in inds1]]
            f, w = super(ActorObserverLossWithClassifier, self).forward(*vars1)
            final.append(f)
            for i, ii in enumerate(inds1):
                weights[ii] = w[i]

        # Classification loss
        if len(inds2) > 0:
            vars2 = var_subset([cls, -target.long()], inds2)
            clsloss = self.clsloss(*vars2)
            f = self.clsweight * clsloss.sum()
            final.append(f)

        final[0], weights = fix(final[0], weights)

        dprint('losses: {}', ' '.join(['{}'.format(r.data[0]) for r in final]))
        return sum(final), weights
