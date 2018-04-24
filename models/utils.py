import torch
import torch.nn as nn
import torchvision.models as tmodels
import torch.distributed as dist
import importlib


VERBOSE = True


def dprint(message, *args):
    if VERBOSE:
        print(message.format(*args))


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


def load_partial_state(model, state_dict):
    # @chenyuntc
    sd = model.state_dict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        if k not in sd or not sd[k].shape == v.shape:
            print('ignoring state key for loading: {}'.format(k))
            continue
        if isinstance(v, torch.nn.Parameter):
            v = v.data
        sd[k].copy_(v)


def remove_last_layer(model):
    # remove last layer
    if hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        model.outdim = newcls[-1].in_features
        model.classifier = nn.Sequential(*newcls[:-1])
    elif hasattr(model, 'fc'):
        model.outdim = model.fc.in_features
        model.fc = IdentityModule()
        # if hasattr(model, 'AuxLogits'):
        #     model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, args.nclass)
    else:
        newcls = list(model.children())[:-1]
        model = nn.Sequential(*newcls[:-1])


def generic_load(arch, pretrained, weights, args):
    if arch in tmodels.__dict__:  # torchvision models
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = tmodels.__dict__[arch](pretrained=True)
            model = model.cuda()
        else:
            print("=> creating model '{}'".format(arch))
            model = tmodels.__dict__[arch]()
    else:  # defined as script in this directory
        model = importlib.import_module('.' + arch, package='models')
        model = model.__dict__[arch](args)

    if not weights == '':
        print('loading pretrained-weights from {}'.format(weights))
        chkpoint = torch.load(weights)
        if isinstance(chkpoint, dict) and 'state_dict' in chkpoint:
            chkpoint = chkpoint['state_dict']
        load_partial_state(model, chkpoint)
    return model


def load_sub_architecture(args):
    model = generic_load(args.subarch, args.pretrained, args.pretrained_subweights, args)
    return model


def load_architecture(args):
    model = generic_load(args.arch, args.pretrained, args.pretrained_weights, args)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        model.cuda()
        model = nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)
        model.cuda()

    return model


def load_criterion(args):
    if hasattr(nn, args.loss):
        criterion = nn.__dict__[args.loss]().cuda()
    else:
        criterion = importlib.import_module('models.layers.' + args.loss)
        criterion = criterion.__dict__[args.loss](args).cuda()
    return criterion
