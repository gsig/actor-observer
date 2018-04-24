"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
import torch
from utils import load_architecture, load_criterion


def create_model(args):
    model = load_architecture(args)
    criterion = load_criterion(args)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    torch.backends.cudnn.benchmark = True
    return model, criterion, optimizer
