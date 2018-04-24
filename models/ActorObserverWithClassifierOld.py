"""
   Use ActorObserver model with classifier output
"""
import torch.nn as nn
from models.ActorObserverBase import ActorObserverBase


class ActorObserverWithClassifierOld(ActorObserverBase):
    def __init__(self, args):
        super(ActorObserverWithClassifierOld, self).__init__(args)
        dim = self.basenet.outdim
        self.classifier = nn.Linear(dim, args.nclass)

    def forward(self, x, y, z):
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
        """
        base_x, base_y, base_z, dist_a, dist_b = self.base(x, y, z)
        w_x = self.firstpos_fc(base_x).view(-1) * (self.firstpos_scale)
        w_y = self.third_fc(base_y).view(-1) * (self.third_scale)
        w_z = self.firstneg_fc(base_z).view(-1) * (self.firstneg_scale)
        cls = self.classifier(base_y)
        self.verbose()
        return dist_a, dist_b, w_x, w_y, w_z, cls
