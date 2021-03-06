""" Define random data for quick debugging """
import torchvision
import torchvision.transforms as transforms


class Fakeego(object):
    @classmethod
    def get(cls, args):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = torchvision.datasets.FakeData(
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: [x, x, x])
            ]))

        val_dataset = torchvision.datasets.FakeData(
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: [x, x, x])
            ]))

        return train_dataset, val_dataset, val_dataset
