import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import shutil
import torch
import torch.nn as nn
import ExampleNetwork

from Trainer import Train, Evaluate
from Dataset import GetDataLoaders
from Utility import Result, StandardOutputDuplicator

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

def TrainWorker(configuration, logFile):
    """
    Worker function for single GPU training
    Components like network, optimizer will be constructed for each process
    """
    sys.stdout = StandardOutputDuplicator(sys.stdout, logFile)

    net = GetNet(configuration).cuda()

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

    trainLoader, validationLoader, testLoader = GetDataLoaders(
        configuration["DatasetName"],
        configuration["BatchSize"],
        configuration["NumOfWorker"],
        configuration["DataFolder"],
        distributed = False
    )

    bestEpoch = 1
    bestIndicator = 0x7FFF
    result = Result()

    for epoch in range(1, configuration["NumOfEpoch"] + 1):
        # Do possible injector for control net during training
        if injector is not None:
            injector(net, epoch)

        # Train one epoch
        trainLoss, trainAccuracy = Train(
            trainLoader, net, optimizer,
            lossFunction, epoch, 0, mode = "single"
        )

        # Do evaluation (some dataset has no validation set)
        if len(validationLoader.dataset) > 0:
            validationLoss, validationAccuracy = Evaluate(
                validationLoader, net, lossFunction,
                "Validation", 0, 0, mode = "single"
            )
        else:
            validationLoss, validationAccuracy = None, None

        # Do evaluation on test set
        # Set testLoss and testAccuracy to None if you don't want to do it every epoch
        testLoss, testAccuracy = Evaluate(
            testLoader, net, lossFunction,
            "Test", 0, 0, mode = "single"
        )

        # Step the learn rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging and saving the best model to the save folder on the main process
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

    print(f"Best epoch -> {bestEpoch}")
    return result
        
def Main(configuration):
    if not os.path.exists(configuration["ResultFolder"]):
        os.mkdir(configuration["ResultFolder"])

    saveFolder = os.path.join(
        configuration["ResultFolder"],
        f"{configuration['NetName']}-{configuration['DatasetName']}-" + time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    )
    os.mkdir(saveFolder)
    configuration["SaveFolder"] = saveFolder

    try:
        logFile = open(os.path.join(configuration["SaveFolder"], "Log.txt"), mode = "w")
        result = TrainWorker(configuration, logFile)
    except Exception as e:
        # If any exception occurs, delete the save folder
        logFile.close()
        shutil.rmtree(configuration["SaveFolder"])
        raise e
    else:
        result.Save(os.path.join(configuration["SaveFolder"], f"Result.txt"))

        # Log the source code
        shutil.copy(
            __file__,
            os.path.join(configuration["SaveFolder"], "TrainScript.py")
        )

if __name__ == "__main__":
    configuration = {
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
