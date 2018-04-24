""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
import torch.utils.data.distributed
import importlib


def case_getattr(obj, attr):
    casemap = {}
    for x in obj.__dict__:
        casemap[x.lower()] = x
    return getattr(obj, casemap[attr.lower()])


def get_dataset(args):
    dataset = importlib.import_module('.' + args.dataset.lower(), package='datasets')
    datasets = case_getattr(dataset, args.dataset).get(args)
    train_dataset, val_dataset, valvideo_dataset = datasets[:3]

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    valvideo_loader = torch.utils.data.DataLoader(
        valvideo_dataset, batch_size=25, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if len(datasets) > 3:
        valvideoego_loader = torch.utils.data.DataLoader(
            datasets[3], batch_size=25, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        return train_loader, val_loader, valvideo_loader, valvideoego_loader
    else:
        return train_loader, val_loader, valvideo_loader
