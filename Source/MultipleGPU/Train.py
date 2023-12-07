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
from Dataset import GetDataLoaders
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

def GetNet(configuration):
    """
    Return the network according to the dataset
    """
    if "MNIST" in configuration["DatasetName"]:
        inputSize = 1024
        outputSize = 10
        parameters = inputSize, outputSize
    elif "CIFAR" in configuration["DatasetName"]:
        inChannels = 3
        numOfClass = int(configuration["DatasetName"].split("CIFAR")[1])
        parameters = inChannels, numOfClass
    elif configuration["DatasetName"] == "TinyImageNet":
        inChannels = 3
        numOfClass = 200
        parameters = inChannels, numOfClass
    else:
        raise ValueError(f"Unknown dataset name {configuration['DatasetName']}")

    net = getattr(ExampleNetwork, configuration["NetName"])(*parameters)
    return net

def DDPTrainWorker(rank, worldSize, sharedDictionary):
    """
    Worker function for parallel training
    Components like network, optimizer will be constructed for each process

    Parameters
    ----------
    rank : int
        The rank of current process, used for logging
    
    worldSize : int
        The number of processes, used for logging

    sharedDictionary : multiprocessing.Manager.dict
        A shared dictionary between all processes, carrying the configuration, save folder and results
    """
    configuration = sharedDictionary["Configuration"]
    if rank == 0:
        logFile = open(os.path.join(configuration["SaveFolder"], "Log.txt"), mode = "w")
        sys.stdout = StandardOutputDuplicator(sys.stdout, logFile)

    SetupEnvironment(rank, worldSize)

    torch.cuda.set_device(rank)
    net = GetNet(configuration).cuda()
    net = nn.parallel.DistributedDataParallel(net, device_ids = [rank])

    lossFunction = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr = configuration["LearnRate"],
        weight_decay = 0.0005,
        momentum = 0.9,
        nesterov = True
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: configuration["LearnRate"] * (0.5 ** (epoch // 10))
    )

    # A injector can be injected into the beginning of each epoch with parameters (net, epoch)
    injector = None

    trainLoader, validationLoader, testLoader, trainSampler = GetDataLoaders(
        configuration["DatasetName"],
        configuration["BatchSize"],
        configuration["NumOfWorker"],
        configuration["DataFolder"],
        distributed = True
    )

    bestEpoch = 1
    bestIndicator = 0x7FFF
    result = Result()

    for epoch in range(1, configuration["NumOfEpoch"] + 1):
        # Do possible injector for control net during training
        if injector is not None:
            injector(net, epoch)

        # Seed for distributed sampler
        trainSampler.set_epoch(epoch)

        # Train one epoch
        trainLoss, trainAccuracy = Train(
            trainLoader, net, optimizer,
            lossFunction, epoch, rank, mode = "multiple"
        )

        # Do evaluation (some dataset has no validation set)
        if len(validationLoader.dataset) > 0:
            validationLoss, validationAccuracy, _ = Evaluate(
                validationLoader, net, lossFunction,
                "Validation", rank, worldSize, mode = "multiple"
            )
        else:
            validationLoss, validationAccuracy = None, None

        # Do evaluation on test set
        # Set testLoss and testAccuracy to None if you don't want to do it every epoch
        testLoss, testAccuracy, _ = Evaluate(
            testLoader, net, lossFunction,
            "Test", rank, worldSize, mode = "multiple"
        )

        # Step the learn rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging and saving the best model to the save folder on the main process
        if rank == 0:
            result.Append(
                trainLoss, trainAccuracy,
                validationLoss, validationAccuracy,
                testLoss, testAccuracy
            )

            if len(validationLoader.dataset) > 0:
                indicator = validationLoss
            else:
                indicator = trainLoss

            if indicator < bestIndicator:
                bestEpoch = epoch
                torch.save(net.state_dict(), os.path.join(configuration["SaveFolder"], f"Weights.pkl"))
            bestIndicator = min(bestIndicator, indicator)

    if rank == 0:
        print(f"Best epoch -> {bestEpoch}")
        sharedDictionary["Result"] = result

    CleanEnvironment()
        
def Main(configuration):
    """
    Spawn processes for the parallel training
    """
    if not os.path.exists(configuration["ResultFolder"]):
        os.mkdir(configuration["ResultFolder"])

    saveFolder = os.path.join(
        configuration["ResultFolder"],
        f"{configuration['NetName']}-{configuration['DatasetName']}-{time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime())}"
    )
    os.mkdir(saveFolder)
    configuration["SaveFolder"] = saveFolder

    try:
        with mp.Manager() as manager:
            sharedDictionary = manager.dict({"Configuration": configuration})

            configuration["BatchSize"] = configuration["BatchSize"] // configuration["NumOfGPU"]
            mp.spawn(
                DDPTrainWorker,
                args = (configuration["NumOfGPU"], sharedDictionary),
                nprocs = configuration["NumOfGPU"]
            )

            sharedDictionary = dict(sharedDictionary)
    except Exception as e:
        # If any exception occurs, delete the save folder
        shutil.rmtree(configuration["SaveFolder"])
        raise e
    else:
        sharedDictionary["Result"].Save(os.path.join(configuration["SaveFolder"], f"Result.txt"))

        # Log the source code
        shutil.copy(
            __file__,
            os.path.join(configuration["SaveFolder"], "TrainScript.py")
        )

if __name__ == "__main__":
    configuration = {
        "NumOfGPU": 2,
        "NumOfWorker": 8,
        "LearnRate": 1e-1,
        "BatchSize": 128,
        "NumOfEpoch": 10,
        "NetName": "FCNN", # should be defined in ExampleNetwork.py
        "DatasetName": "MNIST", # should be able to be recognized by GetDataLoaders in Dataset.py
        "DataFolder": os.path.join(os.path.dirname(__file__), "..", "..", "Data"),
        "ResultFolder": os.path.join(os.path.dirname(__file__), "..", "..", "Result")
    }
    Main(configuration)
