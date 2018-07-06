"""
    Dataset loader that combines Charadesego and Charades
    train and val loaders are concatenated
    Charades includes a [x, x, x] transform to match the input to charadesego
    labels for Charades are negative to distinguish between them
"""
import torchvision.transforms as transforms
from torch.utils.data.dataset import ConcatDataset
from .charadesrgb import Charades
import datasets.charadesego as charadesego
import copy


class CharadesMeta(Charades):
    def __init__(self, *args, **kwargs):
        super(CharadesMeta, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        ims, target, meta = super(CharadesMeta, self).__getitem__(index)
        meta.update({'thirdtime': 0,
                     'firsttime_pos': 0,
                     'firsttime_neg': 0,
                     'n': 0,
                     'n_ego': 0,
                     })
        return ims, target, meta


class CharadesEgoPlusRGB(CharadesMeta):
    @classmethod
    def get(cls, args):
        train_datasetego, val_datasetego, _ = charadesego.CharadesEgo.get(args)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        newargs = copy.deepcopy(args)
        vars(newargs).update({
            'train_file': args.original_charades_train,
            'val_file': args.original_charades_test,
            'data': args.original_charades_data})
        train_dataset, val_dataset, valvideo_dataset = super(CharadesEgoPlusRGB, cls).get(newargs)
        train_dataset.transform.transforms.append(transforms.Lambda(lambda x: [x, x, x]))
        val_dataset.transform.transforms.append(transforms.Lambda(lambda x: [x, x, x]))
        valvideo_dataset.transform.transforms.append(transforms.Lambda(lambda x: [x, x, x]))
        train_dataset.target_transform = transforms.Lambda(lambda x: -x)
        val_dataset.target_transform = transforms.Lambda(lambda x: -x)

        valvideoego_dataset = CharadesMeta(
            args.data, 'val_video',
            args.egocentric_test_data,
            args.cache,
            args.cache_buster,
            transform=transforms.Compose([
                transforms.Resize(int(256. / 224 * args.inputsize)),
                transforms.CenterCrop(args.inputsize),
                transforms.ToTensor(),
                normalize,
            ]))

        train_dataset = ConcatDataset([train_dataset] + [train_datasetego] * 6)
        val_dataset = ConcatDataset([val_dataset] + [val_datasetego] * 6)
        return train_dataset, val_dataset, valvideo_dataset, valvideoego_dataset
