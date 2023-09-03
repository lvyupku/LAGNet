from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.distributed as dist
from .orid import Orid
import torch

def build_transform(is_train):
    t = []
    t.append(transforms.Resize(384))
    t.append(transforms.CenterCrop(384))
    if is_train:
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.RandomVerticalFlip(p=0.5))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    (dataset_val, dataset_test), _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    if dataset_test:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )
    else:
        data_loader_test = data_loader_val

   
    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test


def build_dataset(is_train, config):
    if config.DATA.DATASET == 'cheX':
        #dataset = GetCheXData(config=config, transform=transform, isTrain=is_train)
        nb_classes = 13
    elif config.DATA.DATASET == 'nih':
        #dataset = GetNIHData(config=config, transform=transform, isTrain=is_train)
        nb_classes = 14
    elif config.DATA.DATASET == 'orid':
        if is_train:
            dataset = Orid('dataset/ODIR-5K_Training_Dataset', 'dataset/train1.xlsx', build_transform(is_train))
        else:
            valid = Orid('dataset/ODIR-5K_Training_Dataset', 'dataset/valid1.xlsx', build_transform(is_train))
            test = Orid('dataset/ODIR-5K_Training_Dataset', 'dataset/test1.xlsx', build_transform(is_train))
            dataset = (valid, test)
        nb_classes = 7
    else:
        raise NotImplementedError("dataset error.")

    return dataset, nb_classes
