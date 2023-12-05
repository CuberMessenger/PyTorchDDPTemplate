import os
import json
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ImageNetDataset(Dataset):
    """
    This is a modified dataset class for ImageNet
    """
    def __init__(self, root, split, transform = None):
        self.Samples = []
        self.Targets = []
        self.Transform = transform
        self.SynToClass = {}

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as jsonFile:
            jsonData = json.load(jsonFile)
            for classID, v in jsonData.items():
                self.SynToClass[v[0]] = int(classID)

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as jsonFile:
            self.ValToSyn = json.load(jsonFile)
    
        sampleFolders = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(sampleFolders):
            if split == "train":
                synID = entry
                target = self.SynToClass[synID]
                synFolder = os.path.join(sampleFolders, synID)
                for sample in os.listdir(synFolder):
                    samplePath = os.path.join(synFolder, sample)
                    self.Samples.append(samplePath)
                    self.Targets.append(target)
            elif split == "val":
                synID = self.ValToSyn[entry]
                target = self.SynToClass[synID]
                samplePath = os.path.join(sampleFolders, entry)
                self.Samples.append(samplePath)
                self.Targets.append(target)

    def __len__(self):
        return len(self.Samples)

    def __getitem__(self, idx):
        x = Image.open(self.Samples[idx]).convert("RGB")
        if self.Transform:
            x = self.Transform(x)
        return x, self.Targets[idx]


def GetDataLoaders(datasetName, batchSize, numOfWorker, saveFolder, distributed = False):
    """
    Return train, validation and test data loaders
    If no validation set is available, the validation loader is a loader object with no data
    
    Parameters
    ----------
    datasetName : str
        MNIST, CIFAR10, CIFAR100, ImageNet, TinyImageNet

    batchSize : int

    numOfWorker : int

    saveFolder : str
        Path to the folder to save the datasets
        A default folder is used for MNIST and CIFAR
        Folders of ImageNet and TinyImageNet should be specified in the configuration

    distributed : bool, default False
        If True, the data loaders are created with distributed samplers
    """
    if "MNIST" in datasetName:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.25)
        ])

        trainSet = datasets.MNIST(root = saveFolder, train = True, transform = transform, download = True)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [50000, 10000])
        testSet = datasets.MNIST(root = saveFolder, train = False, transform = transform, download = True)
    elif "CIFAR" in datasetName:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        trainTransform = transforms.Compose([
            transforms.RandomAffine(
                degrees = 15,
                translate = (0.1, 0.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        evaluationTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainSet = getattr(datasets, datasetName)(root = saveFolder, train = True, transform = trainTransform, download = True)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [50000, 0])
        testSet = getattr(datasets, datasetName)(root = saveFolder, train = False, transform = evaluationTransform, download = True)
    elif datasetName == "ImageNet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainSet = ImageNetDataset(root = saveFolder, split = "train", transform = transform)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [1281167, 0])
        testSet = ImageNetDataset(root = saveFolder, split = "val", transform = transform)
    elif datasetName == "TinyImageNet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        trainTransform = transforms.Compose([
            transforms.RandomAffine(
                degrees = 10,
                translate = (0.1, 0.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainSet = datasets.ImageFolder(root = os.path.join(saveFolder, "train"), transform = trainTransform)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [100000, 0])
        testSet = datasets.ImageFolder(root = os.path.join(saveFolder, "val"), transform = testTransform)
    else:
        raise Exception(f"Unknown dataset name {datasetName}")

    """
    On Windows, the data loading processes cost lots of time to start for every epoch
    Persistent workers may save the initialization time after the first epoch
    """
    persistentWorkers = numOfWorker > 0
    trainSampler, validationSampler, testSampler = None, None, None
    if distributed:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainSet)
        validationSampler = torch.utils.data.distributed.DistributedSampler(validationSet, shuffle = False)
        testSampler = torch.utils.data.distributed.DistributedSampler(testSet, shuffle = False)
    trainLoader = torch.utils.data.DataLoader(
        trainSet, batch_size = batchSize, shuffle = not distributed, sampler = trainSampler,
        num_workers = numOfWorker, persistent_workers = persistentWorkers, pin_memory = True
    )
    validationLoader = torch.utils.data.DataLoader(
        validationSet, batch_size = batchSize, sampler = validationSampler,
        num_workers = numOfWorker, persistent_workers = persistentWorkers, pin_memory = True
    )
    testLoader = torch.utils.data.DataLoader(
        testSet, batch_size = batchSize, sampler = testSampler,
        num_workers = numOfWorker, persistent_workers = persistentWorkers, pin_memory = True
    )

    if distributed:
        return trainLoader, validationLoader, testLoader, trainSampler
    else:
        return trainLoader, validationLoader, testLoader

if __name__ == "__main__":
    train, val, test = GetDataLoaders("TinyImageNet", 100, 10, "../tiny-imagenet-200")
    for batchX, batchY in train:
        print(batchX.shape)
        print(batchY.shape)
        break