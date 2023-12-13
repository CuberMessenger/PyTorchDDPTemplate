import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import shutil
import argparse
import torch
import torch.nn as nn
import Network

from Trainer import Train, Evaluate
from Dataset import GetDataLoaders
from Utility import Result, StandardOutputDuplicator
from Train import GetNet

def TestWorker(configuration, logFile):
    """
    Worker function for single GPU testing
    """
    sys.stdout = StandardOutputDuplicator(sys.stdout, logFile)

    net = GetNet(configuration).cuda()
    net.load_state_dict(torch.load(os.path.join(configuration["WeightFolder"], "Weights.pkl")))

    lossFunction = nn.CrossEntropyLoss().cuda()

    _, _, testLoader = GetDataLoaders(
        configuration["DatasetName"],
        configuration["BatchSize"],
        configuration["NumOfWorker"],
        configuration["DataFolder"],
        distributed = False
    )

    testLoss, testAccuracy, testPredictions = Evaluate(
        testLoader, net, lossFunction,
        "Test", 0, 0, mode = "single"
    )

    results = {
        "TestLoss": testLoss,
        "TestAccuracy": testAccuracy,
        "TestPredictions": testPredictions
    }

    torch.save(results, os.path.join(configuration["SaveFolder"], "TestResults.pt"))

def Main(configuration):
    if not os.path.exists(configuration["ResultFolder"]):
        os.mkdir(configuration["ResultFolder"])

    folderName = os.path.basename(configuration["WeightFolder"])
    if folderName == "":
        # If the parsed path looks like "a/b/c/", the basename will be ""
        folderName = os.path.basename(os.path.dirname(configuration["WeightFolder"]))

    saveFolder = os.path.join(
        configuration["ResultFolder"],
        f"{folderName}-Test-{time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime())}"
    )
    os.mkdir(saveFolder)
    configuration["SaveFolder"] = saveFolder

    try:
        with open(os.path.join(configuration["SaveFolder"], "Log.txt"), mode = "w") as logFile:
            TestWorker(configuration, logFile)
    except Exception as e:
        # If any exception occurs, delete the save folder
        shutil.rmtree(configuration["SaveFolder"])
        raise e
    else:
        # Log the source code
        shutil.copy(
            __file__,
            os.path.join(configuration["SaveFolder"], "TestScript.py")
        )

if __name__ == "__main__":
    configuration = {
        "NumOfWorker": 8,
        "LearnRate": 1e-1,
        "BatchSize": 128,
        "NumOfEpoch": 10,
        "NetName": "FCNN", # should be defined in Network.py
        "DatasetName": "MNIST", # should be able to be recognized by GetDataLoaders in Dataset.py
        "DataFolder": os.path.join(os.path.dirname(__file__), "..", "..", "Data"),
        "ResultFolder": os.path.join(os.path.dirname(__file__), "..", "..", "Result"),
        "WeightFolder": None # path to a folder containing a Weights.pkl file
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-folder", type = str)
    args = parser.parse_args()

    configuration["WeightFolder"] = args.weight_folder

    Main(configuration)
