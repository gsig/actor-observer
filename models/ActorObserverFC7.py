"""
   Use ActorObserver model as first person classifier
"""
import torch
from models.ActorObserverBase import ActorObserverBase
from models.utils import dprint


class ActorObserverFC7(ActorObserverBase):
    def __init__(self, args):
        if isinstance(args, dict):
            super(ActorObserverFC7, self).__init__(args)
        else:
            if 'DataParallel' in args.__class__.__name__:
                args = args.module
            print('Initializing FC7 extractor with AOB instance')
            self.__dict__ = args.__dict__

    def forward(self, x, y, z):
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
        """
        base_x = self.basenet(x)
        base_y = self.basenet(y)
        w_x = self.firstpos_fc(base_x).view(-1) * torch.exp(self.firstpos_scale)
        w_y = self.third_fc(base_x).view(-1) * torch.exp(self.third_scale)
        dprint('fc7 norms: {}\t {}', base_x.data.norm(), base_y.data.norm())
        self.verbose()
        return base_x, base_y, w_x, w_y
