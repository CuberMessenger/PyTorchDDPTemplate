import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import shutil
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import Network

from Trainer import Train, Evaluate
from Dataset import GetDataLoaders
from Utility import Result, StandardOutputDuplicator, MemoryUsage
from Train import SetupEnvironment, CleanEnvironment, GetNet

def DDPTestWorker(rank, worldSize, sharedDictionary):
    """
    Worker function for parallel testing
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
        sys.stdout = StandardOutputDuplicator(logFile)

    SetupEnvironment(rank, worldSize)

    torch.cuda.set_device(rank)
    net = GetNet(configuration).cuda()
    net = nn.parallel.DistributedDataParallel(net, device_ids = [rank])

    net.load_state_dict(sharedDictionary["Weight"])
    lossFunction = nn.CrossEntropyLoss().cuda()

    _, _, testLoader, _ = GetDataLoaders(
        configuration["DatasetName"],
        configuration["BatchSize"],
        configuration["NumOfWorker"],
        configuration["DataFolder"],
        distributed = True
    )

    testLoss, testAccuracy, testPredictions = Evaluate(
        testLoader, net, lossFunction,
        "Test", rank, worldSize, mode = "multiple"
    )

    if rank == 0:
        results = {
            "TestLoss": testLoss,
            "TestAccuracy": testAccuracy,
            "TestPredictions": testPredictions
        }

        torch.save(results, os.path.join(configuration["SaveFolder"], "TestResults.pt"))

    CleanEnvironment()
        
def Main(configuration):
    """
    Spawn processes for the parallel testing
    """
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
        with mp.Manager() as manager:
            sharedDictionary = manager.dict({
                "Configuration": configuration,
                "Weight": torch.load(os.path.join(configuration["WeightFolder"], "Weights.pkl"), map_location = "cpu")
            })
            """
            It is important to load the weights to cpu
            Otherwise, the weights will be loaded to the device it was saved from (normally GPU 0)
            It will cost non-negligible GPU memory and will not be collected automatically

            Also, the weight is loaded here in the main process to avoid duplicating it in CPU memory
            """

            configuration["BatchSize"] = configuration["BatchSize"] // configuration["NumOfGPU"]

            mp.spawn(
                DDPTestWorker,
                args = (configuration["NumOfGPU"], sharedDictionary),
                nprocs = configuration["NumOfGPU"]
            )

            sharedDictionary = dict(sharedDictionary)
    except Exception as e:
        # If any exception occurs, delete the save folder
        shutil.rmtree(configuration["SaveFolder"])
        raise e
    else:
        # Log the source code
        shutil.copy(
            __file__,
            os.path.join(configuration["SaveFolder"], "TrainScript.py")
        )

if __name__ == "__main__":
    configuration = {
        "NumOfGPU": None,
        "NumOfWorker": 8,
        "LearnRate": 1e-1,
        "BatchSize": 128,
        "NumOfEpoch": 10,
        "NetName": "FCNN", # should be defined in Network.py
        "DatasetName": "MNIST", # should be able to be recognized by GetDataLoaders in Dataset.py
        "DataFolder": os.path.join(os.path.dirname(__file__), "..", "..", "Data"),
        "ResultFolder": os.path.join(os.path.dirname(__file__), "..", "..", "Result"),
        "WeightFolder": None
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-of-gpu", type = int)
    parser.add_argument("--weight-folder", type = str)
    args = parser.parse_args()

    if args.num_of_gpu is None:
        configuration["NumOfGPU"] = torch.cuda.device_count()
    else:
        configuration["NumOfGPU"] = args.num_of_gpu
    configuration["WeightFolder"] = args.weight_folder

    Main(configuration)
