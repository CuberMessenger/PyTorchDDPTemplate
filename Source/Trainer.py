import time
import torch

from Utility import AverageMeter, TopKAccuracy

def Train(trainLoader, net, optimizer, lossFunction, epoch, rank, mode = "multiple"):
    """
    Train one epoch with given data loader, network, optimizer and loss function

    Parameters
    ----------
    trainLoader : torch.utils.data.DataLoader

    net : torch.nn.Module
        Should be on cuda device already
    
    optimizer : torch.optim.Optimizer

    lossFunction : torch.nn.Module

    epoch : int
        The current epoch, used for logging

    rank : int
        The rank of current process, used for logging

    mode : str, default = "multiple"
        multiple or single
    """
    batchTime = AverageMeter("BatchTime")
    dataTime = AverageMeter("DataTime")
    losses = AverageMeter("Losses")
    top1Accuracies = AverageMeter("Top1Accuracy")
    top5Accuracies = AverageMeter("Top5Accuracy")

    net.train()

    startTime = time.perf_counter_ns()
    for batchData, batchLabel in trainLoader:
        dataTime.Update((time.perf_counter_ns() - startTime) / 1e6)

        batchData = batchData.cuda()
        batchLabel = batchLabel.cuda()

        batchPredict = net(batchData)
        loss = lossFunction(batchPredict, batchLabel)

        losses.Update(loss.item(), batchData.size(0))
        top1accuracy = TopKAccuracy(batchPredict, batchLabel, topk = (1,))
        top5accuracy = TopKAccuracy(batchPredict, batchLabel, topk = (5,))
        top1Accuracies.Update(top1accuracy[0], batchData.size(0))
        top5Accuracies.Update(top5accuracy[0], batchData.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchTime.Update((time.perf_counter_ns() - startTime) / 1e6)

    if mode == "multiple":
        batchTime.AllReduce()
        dataTime.AllReduce()
        losses.AllReduce()
        top1Accuracies.AllReduce()
        top5Accuracies.AllReduce()

    if mode == "single":
        top1Accuracies.Average = top1Accuracies.Average.item()
        top5Accuracies.Average = top5Accuracies.Average.item()

    if rank == 0 or mode == "single":
        toPrint = f"\nEpoch [{epoch}]\n"
        toPrint += f"Train:      [loss, accuray] -> [{losses.Average:.4e}, ({top1Accuracies.Average:.2f}%, {top5Accuracies.Average:.2f}%)], "
        toPrint += f"[bacth time, data time] -> [{batchTime.Average:.6f}ms, {dataTime.Average:.6f}ms]"
        print(toPrint)

    return losses.Average, top1Accuracies.Average

def Evaluate(testLoader, net, lossFunction, name, rank, worldSize, mode = "multiple"):
    """
    Evaluate the network with given data loader and loss function

    Parameters
    ----------
    testLoader : torch.utils.data.DataLoader

    net : torch.nn.Module
        Should be on cuda device already

    lossFunction : torch.nn.Module

    name : str
        Should be "Test" or "Validation"

    rank : int
        The rank of current process, used for logging

    worldSize : int
        The number of processes, used for logging

    mode : str, default = "multiple"
        multiple or single
    """
    def EvaluateLoop(loader):
        with torch.no_grad():
            startTime = time.perf_counter_ns()
            for batchData, batchLabel in loader:
                batchData = batchData.cuda()
                batchLabel = batchLabel.cuda()

                batchPredict = net(batchData)
                loss = lossFunction(batchPredict, batchLabel)

                losses.Update(loss.item(), batchData.size(0))
                top1accuracy = TopKAccuracy(batchPredict, batchLabel, topk = (1,))
                top5accuracy = TopKAccuracy(batchPredict, batchLabel, topk = (5,))
                top1Accuracies.Update(top1accuracy[0], batchData.size(0))
                top5Accuracies.Update(top5accuracy[0], batchData.size(0))

                batchTime.Update((time.perf_counter_ns() - startTime) / 1e6)

    batchTime = AverageMeter("BatchTime")
    losses = AverageMeter("Losses")
    top1Accuracies = AverageMeter("Top1Accuracy")
    top5Accuracies = AverageMeter("Top5Accuracy")

    net.eval()
    EvaluateLoop(testLoader)

    if mode == "multiple":
        if len(testLoader.sampler) * worldSize < len(testLoader.dataset):
            auxiliaryTestDataset = torch.utils.data.Subset(
                testLoader.dataset,
                range(len(testLoader.sampler) * worldSize, len(testLoader.dataset))
            )
            auxiliaryTestLoader = torch.utils.data.DataLoader(
                auxiliaryTestDataset, batch_size = testLoader.batch_size, shuffle = False,
                num_workers = testLoader.num_workers, persistent_workers = testLoader.persistent_workers, pin_memory = True
            )
            EvaluateLoop(auxiliaryTestLoader, jump = len(testLoader))

        batchTime.AllReduce()
        losses.AllReduce()
        top1Accuracies.AllReduce()
        top5Accuracies.AllReduce()

    if mode == "single":
        top1Accuracies.Average = top1Accuracies.Average.item()
        top5Accuracies.Average = top5Accuracies.Average.item()

    if rank == 0 or mode == "single":
        toPrint = ""
        toPrint += f"{name}:{' ' if len(name) > 4 else '       '}[loss, accuray] -> [{losses.Average:.4e}, ({top1Accuracies.Average:.2f}%, {top5Accuracies.Average:.2f}%)], "
        toPrint += f"bacth time -> {batchTime.Average:.6f}ms"
        print(toPrint)
        
    return losses.Average, top1Accuracies.Average
