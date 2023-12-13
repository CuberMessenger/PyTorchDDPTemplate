import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import shutil
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import ExampleNetwork

from Trainer import Train, Evaluate
from Dataset import GetDataLoaders, DatasetWithIdentity
from Utility import Result, StandardOutputDuplicator

def SetupEnvironment(rank, worldSize):
    """
    Initialize the environment for distributed training
    The port number should be free on the machine
    Should be called at the start of the worker process
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "31428"

    torch.distributed.init_process_group(
        backend = "nccl",
        rank = rank,
        world_size = worldSize
    )

    torch.cuda.set_device(rank)

def CleanEnvironment():
    """
    Clean the environment for distributed training
    Should be called at the end of the worker process
    """
    torch.distributed.destroy_process_group()

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length, size, classes):
        self.X = torch.ones((length, size))
        for i in range(length):
            self.X[i] *= i
        self.Y = torch.randint(classes - 1, (length,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def Worker(rank, worldSize):
    SetupEnvironment(rank, worldSize)

    net = nn.Linear(20, 10).cuda()
    net = nn.parallel.DistributedDataParallel(net, device_ids = [rank])
    lossFunction = nn.CrossEntropyLoss().cuda()

    dataset = DummyDataset(33, 20, 10)
    dataset = DatasetWithIdentity(dataset)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = False, drop_last = True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = 5, sampler = sampler,# drop_last = False,
        num_workers = 2, persistent_workers = True, pin_memory = True
    )
    print(f"Sampler drop_last: {sampler.drop_last}")
    print(f"Loader drop_last: {loader.drop_last}")

    testLoss, testAccuracy, testPredictions = Evaluate(
        loader, net, lossFunction,
        "Test", rank, worldSize, mode = "multiple"
    )

    if rank == 0:
        print(testPredictions.size())

        for i, p in enumerate(testPredictions):
            print(f"{i}: {p}")

    CleanEnvironment()

def Main():
    numOfGPU = 2
    mp.spawn(
        Worker,
        args = (numOfGPU,),
        nprocs = numOfGPU
    )

if __name__ == "__main__":
    Main()


