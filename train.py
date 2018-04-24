""" Defines the Trainer class which handles train/validation/validation_video
"""
import torch
import itertools
import numpy as np
import pdb
from utils import map as meanap
from utils.utils import dump_gpumem, AverageMeter, submission_file, Timer


def adjust_learning_rate(startlr, decay_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = startlr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse(x):
    return x[0], x[1], x[2] if len(x) > 2 else {'id': [1] * x[1].shape[0]}


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def triplet_accuracy(output, target, weights=None):
    """
       if target>0 then first output should be smaller than right output
       optional weighted average
    """
    if type(output) is not list: 
        output = [(x.data[0], y.data[0]) for x, y in zip(*output)]
    correct = [x < y if t > 0 else y < x for (x, y), t in zip(output, target)]
    if weights is None:
        return np.mean(correct)
    else:
        weights = weights.numpy()
        weights = weights / (1e-5 + np.sum(weights))
        return np.sum(np.array(correct).astype(float) * weights)


def triplet_topk(output, target, weights, topk=5):
    weights = np.array(weights)
    n = weights.shape[0]
    topkn = int(np.ceil(.01 * topk * n))
    ind = np.argsort(weights)
    ind = ind[-topkn:].tolist()
    return triplet_accuracy([output[x] for x in ind], [target[x] for x in ind])


def triplet_allk(output, target, weights):
    out = {}
    for k in (1, 2, 5, 10, 50):
        out['topk{}'.format(k)] = triplet_topk(output, target, weights, topk=k)
    return out


def forward(inputs, target, model, criterion, ids, train=True):
    target = target.float().cuda(async=True)
    input_vars = [torch.autograd.Variable(inp.cuda(), volatile=not train)
                  for inp in inputs]
    target_var = torch.autograd.Variable(target, volatile=not train)
    output = model(*input_vars)
    loss, weights = criterion(*(list(output) + [target_var, ids]))
    return output[:2], loss, weights


class Trainer():
    def train(self, loader, model, criterion, optimizer, epoch, args):
        adjust_learning_rate(args.lr, args.lr_decay_rate, optimizer, epoch)
        timer = Timer()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        wtop1 = AverageMeter()
        metrics = {}

        # switch to train mode
        model.train()
        optimizer.zero_grad()

        def part(x):
            return itertools.islice(x, int(len(x) * args.train_size))
        for i, x in enumerate(part(loader)):
            inputs, target, meta = parse(x)
            data_time.update(timer.thetime() - timer.end)
            output, loss, weights = forward(inputs, target, model, criterion, meta['id'])
            prec1 = triplet_accuracy(output, target)
            wprec1 = triplet_accuracy(output, target, weights)
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            wtop1.update(wprec1, inputs[0].size(0))

            loss.backward()
            if i % args.accum_grad == args.accum_grad - 1:
                print('updating parameters')
                optimizer.step()
                optimizer.zero_grad()

            timer.tic()
            if i % args.print_freq == 0:
                print('[{name}] Epoch: [{0}][{1}/{2}({3})]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'WAcc@1 {wtop1.val:.3f} ({wtop1.avg:.3f})\t'.format(
                          epoch, i, int(len(loader) * args.train_size), len(loader), name=args.name,
                          timer=timer, data_time=data_time, loss=losses, top1=top1, wtop1=wtop1))

        metrics.update({'top1': top1.avg, 'wtop1': wtop1.avg})
        return metrics

    def validate(self, loader, model, criterion, epoch, args):
        timer = Timer()
        losses = AverageMeter()
        top1 = AverageMeter()
        wtop1 = AverageMeter()
        alloutputs = []
        metrics = {}

        # switch to evaluate mode
        model.eval()

        def part(x):
            return itertools.islice(x, int(len(x) * args.val_size))
        for i, x in enumerate(part(loader)):
            inputs, target, meta = parse(x)
            output, loss, weights = forward(inputs, target, model, criterion, meta['id'], train=False)
            prec1 = triplet_accuracy(output, target)
            wprec1 = triplet_accuracy(output, target, weights)
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            wtop1.update(wprec1, inputs[0].size(0))
            alloutputs.extend(zip([(x.data[0], y.data[0]) for x, y in zip(*output)], target, weights))
            timer.tic()

            if i % args.print_freq == 0:
                print('[{name}] Test [{epoch}]: [{0}/{1} ({2})]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'WAcc@1 {wtop1.val:.3f} ({wtop1.avg:.3f})\t'.format(
                          i, int(len(loader) * args.val_size), len(loader), name=args.name,
                          timer=timer, loss=losses, top1=top1, epoch=epoch, wtop1=wtop1))

        metrics.update(triplet_allk(*zip(*alloutputs)))
        metrics.update({'top1val': top1.avg, 'wtop1val': wtop1.avg})
        print(' * Acc@1 {top1val:.3f} \t WAcc@1 {wtop1val:.3f}'
              '\n   topk1: {topk1:.3f} \t topk2: {topk2:.3f} \t '
              'topk5: {topk5:.3f} \t topk10: {topk10:.3f} \t topk50: {topk50:.3f}'
              .format(**metrics))

        return metrics

    def validate_video(self, loader, model, epoch, args):
        """ Run video-level validation on the Charades test set"""
        timer = Timer()
        outputs, gts, ids = [], [], []
        metrics = {}

        # switch to evaluate mode
        model.eval()

        for i, x in enumerate(loader):
            inputs, target, meta = parse(x)
            target = target.long().cuda(async=True)
            assert target[0, :].eq(target[1, :]).all(), "val_video not synced"
            input_vars = [torch.autograd.Variable(inp.cuda(), volatile=True)
                          for inp in inputs]
            output = model(*input_vars)[-1]  # classification should be last output
            output = torch.nn.Softmax(dim=1)(output)

            # store predictions
            output_video = output.mean(dim=0)
            outputs.append(output_video.data.cpu().numpy())
            gts.append(target[0, :])
            ids.append(meta['id'][0])
            timer.tic()

            if i % args.print_freq == 0:
                print('Test2: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                          i, len(loader), timer=timer))
        # mAP, _, ap = meanap.map(np.vstack(outputs), np.vstack(gts))
        mAP, _, ap = meanap.charades_map(np.vstack(outputs), np.vstack(gts))
        metrics['mAP'] = mAP
        print(ap)
        print(' * mAP {:.3f}'.format(mAP))
        submission_file(
            ids, outputs, '{}/epoch_{:03d}.txt'.format(args.cache, epoch + 1))
        return metrics

    def validate_egovideo(self, loader, model, epoch, args):
        """ Run video-level validation on the Charades ego test set"""
        timer = Timer()
        outputs, gts, ids = [], [], []
        outputsw = []
        metrics = {}

        # switch to evaluate mode
        model.eval()
        for i, x in enumerate(loader):
            inp, target, meta = parse(x)
            target = target.long().cuda(async=True)
            assert target[0, :].eq(target[1, :]).all(), "val_video not synced"
            input_var = torch.autograd.Variable(inp.cuda(), volatile=True)
            output, w_x, w_z = model(input_var)
            output = torch.nn.Softmax(dim=1)(output)

            sw_x = torch.nn.Softmax(dim=0)(w_x) * w_x.shape[0]
            sw_x = (sw_x - sw_x.mean()) / sw_x.std()
            scale = torch.clamp(1 + (sw_x - 1) * 0.05, 0, 100)
            print('scale min: {}\t max: {}\t std: {}'.format(scale.min().data[0], scale.max().data[0], scale.std().data[0]))
            scale = torch.clamp(scale, 0, 100)
            scale *= scale.shape[0] / scale.sum()
            outputw = output * scale.unsqueeze(1)

            # store predictions
            output_video = output.mean(dim=0)
            outputs.append(output_video.data.cpu().numpy())
            outputsw.append(outputw.mean(dim=0).data.cpu().numpy())
            gts.append(target[0, :])
            ids.append(meta['id'][0])
            timer.tic()

            if i % args.print_freq == 0:
                print('Test2: [{0}/{1}]\t'
                      'Time {timer.val:.3f} ({timer.avg:.3f})'.format(
                          i, len(loader), timer=timer))
        # mAP, _, ap = meanap.map(np.vstack(outputs), np.vstack(gts))
        mAP, _, ap = meanap.charades_nanmap(np.vstack(outputs), np.vstack(gts))
        mAPw, _, _ = meanap.charades_nanmap(np.vstack(outputsw), np.vstack(gts))
        metrics['mAPego'] = mAP
        metrics['mAPegow'] = mAPw
        print(ap)
        print(' * mAPego {mAPego:.3f} \t mAPegow {mAPegow:.3f}'.format(**metrics))
        submission_file(
            ids, outputs, '{}/egoepoch_{:03d}.txt'.format(args.cache, epoch + 1))
        return metrics
