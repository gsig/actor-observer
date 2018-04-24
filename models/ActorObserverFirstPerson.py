"""
   Use ActorObserver model as first person classifier
"""
from models.ActorObserverWithClassifier import ActorObserverWithClassifier
from models.utils import dprint
import torch


class ActorObserverFirstPerson(ActorObserverWithClassifier):
    def __init__(self, args):
        if isinstance(args, dict):
            super(ActorObserverFirstPerson, self).__init__(args)
        else:
            if 'DataParallel' in args.__class__.__name__:
                args = args.module
            print('Initializing FirstPerson classifier with AOWC instance')
            self.__dict__ = args.__dict__

    def forward(self, x):
        """ assuming:
            x: first person positive
            y: third person
            z: first person negative
        """
        base_x = self.basenet(x)
        y = self.classifier(base_x)
        w_x = self.firstpos_fc(base_x).view(-1) * torch.exp(self.firstpos_scale)
        w_z = self.firstneg_fc(base_x).view(-1) * torch.exp(self.firstneg_scale)
        dprint('fc7 norms: {}', base_x.data.norm())
        self.verbose()
        return y, w_x, w_z

