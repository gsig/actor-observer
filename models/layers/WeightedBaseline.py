from ActorObserverLoss import ActorObserverLoss


class WeightedBaseline(ActorObserverLoss):
    def __init__(self, args):
        super(WeightedBaseline, self).__init__(args)

    def forward(self, *args, **kwargs):
        output, weights = super(WeightedBaseline, self).forward(*args, **kwargs)
        weights = (args[1].data.cpu() - args[0].data.cpu()).abs()
        return output, weights
