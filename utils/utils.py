import torch
import gc
import time
import numpy as np


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))


def dump_gpumem():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MedianMeter(AverageMeter):
    """Computes median and stores the average and current value"""

    def __init__(self):
        super(MedianMeter, self).__init__()
        self.vals = []
        self.med = np.nan

    def singleupdate(self, val, n=1):
        super(MedianMeter, self).update(val, n)
        self.vals.append(val)
        self.med = np.median(self.vals)

    def multipleupdate(self, val, n=1):
        for v in val:
            self.singleupdate(v, n)

    def update(self, val, n=1):
        if isinstance(val, int) or isinstance(val, float):
            self.singleupdate(val, n)
        else:
            self.multipleupdate(val, n)


class Timer(AverageMeter):
    def __init__(self):
        super(Timer, self).__init__()
        self.end = time.time()

    def tic(self):
        self.update(time.time() - self.end)
        self.end = time.time()

    def thetime(self):
        return time.time()
