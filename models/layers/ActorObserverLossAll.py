from models.layers.VideoSoftmax import VideoSoftmax
from ActorObserverLoss import ActorObserverLoss


class ActorObserverLossAll(ActorObserverLoss):
    def __init__(self, args):
        super(ActorObserverLossAll, self).__init__(args)

    def get_constants(self, ids):
        same = ['all' for x in ids]
        return super(ActorObserverLossAll, self).get_constants(same)

    def update_constants(self, input, weights, ids):
        same = ['all' for x in ids]
        return super(ActorObserverLossAll, self).update_constants(input, weights, same)

