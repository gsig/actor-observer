"""
   ActorObserver base model without final FC layers shared
"""
from models.ActorObserverBase import ActorObserverBase
from torch import nn


class ActorObserverBaseNoShare(ActorObserverBase):
    def __init__(self, args):
        super(ActorObserverBaseNoShare, self).__init__(args)
        self.firstneg_fc = nn.Sequential(nn.Linear(self.basenet.outdim, 1), nn.Tanh())



