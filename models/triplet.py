"""
Define a triplet model
"""
import torch.nn as nn
import torch.nn.functional as F
from models.utils import load_sub_architecture, remove_last_layer


class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2).view(-1)
        dist_b = F.pairwise_distance(embedded_y, embedded_z, 2).view(-1)
        return dist_a, dist_b


class Triplet(TripletNet):
    def __init__(self, args):
        model = load_sub_architecture(args)
        remove_last_layer(model)
        super(Triplet, self).__init__(model)
