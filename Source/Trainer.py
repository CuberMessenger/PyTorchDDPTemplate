import time
import torch

from Utility import AverageMeter, TopKAccuracy


def Train(trainLoader, net, optimizer, lossFunction, epoch, rank, mode="multiple"):
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
    for batchData, batchLabel, _ in trainLoader:
        dataTime.Update((time.perf_counter_ns() - startTime) / 1e6)

        batchData = batchData.cuda()
        batchLabel = batchLabel.cuda()

        batchPredict = net(batchData)
        loss = lossFunction(batchPredict, batchLabel)

        losses.Update(loss.item(), batchData.size(0))
        top1accuracy = TopKAccuracy(batchPredict, batchLabel, topk=(1,))
        top5accuracy = TopKAccuracy(batchPredict, batchLabel, topk=(5,))
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


def EvaluateSingle(testLoader, net, lossFunction, name):
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
    """
    batchTime = AverageMeter("BatchTime")
    losses = AverageMeter("Losses")
    top1Accuracies = AverageMeter("Top1Accuracy")
    top5Accuracies = AverageMeter("Top5Accuracy")

    def EvaluateLoop(loader):
        batchPredictions = []
        with torch.no_grad():
            startTime = time.perf_counter_ns()
            for batchData, batchLabel, _ in loader:
                batchData = batchData.cuda()
                batchLabel = batchLabel.cuda()

                # print(f"Test process got {len(batchData)} samples!")

                batchPrediction = net(batchData)
                loss = lossFunction(batchPrediction, batchLabel)

                losses.Update(loss.item(), batchData.size(0))
                top1accuracy = TopKAccuracy(batchPrediction, batchLabel, topk=(1,))
                top5accuracy = TopKAccuracy(batchPrediction, batchLabel, topk=(5,))
                top1Accuracies.Update(top1accuracy[0], batchData.size(0))
                top5Accuracies.Update(top5accuracy[0], batchData.size(0))

                batchTime.Update((time.perf_counter_ns() - startTime) / 1e6)

                # Here, the loader is assumed to be not shuffled
                # If shuffled or any disorder appears, you should use the index (the third return value of the loader)
                batchPredictions.append(batchPrediction)

        return batchPredictions

    net.eval()
    batchPredictions = EvaluateLoop(testLoader)

    """
    In single GPU mode, as long as the drop_last of DataLoader is set to False
    The above loop should handle all samples in the dataset, for example:
    15 data samples in the dataloader and batch size is 6, then the loop will be run 3 times with:
    Batch 0: 6 samples
    Batch 1: 6 samples
    Batch 2: 3 samples
    """
    predictions = torch.cat(batchPredictions, dim=0)

    if hasattr(top1Accuracies.Average, "item"):
        top1Accuracies.Average = top1Accuracies.Average.item()
    if hasattr(top5Accuracies.Average, "item"):
        top5Accuracies.Average = top5Accuracies.Average.item()

    toPrint = ""
    toPrint += f"{name}:{' ' if len(name) > 4 else '       '}[loss, accuray] -> [{losses.Average:.4e}, ({top1Accuracies.Average:.2f}%, {top5Accuracies.Average:.2f}%)], "
    toPrint += f"bacth time -> {batchTime.Average:.6f}ms"
    print(toPrint)

    return losses.Average, top1Accuracies.Average, predictions


def EvaluateMultiple(testLoader, net, lossFunction, name, rank, worldSize):
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
    """
    batchTime = AverageMeter("BatchTime")
    losses = AverageMeter("Losses")
    top1Accuracies = AverageMeter("Top1Accuracy")
    top5Accuracies = AverageMeter("Top5Accuracy")

    def EvaluateLoop(loader):
        batchIndexes = []
        batchPredictions = []
        with torch.no_grad():
            startTime = time.perf_counter_ns()
            for batchData, batchLabel, batchIndex in loader:
                batchData = batchData.cuda()
                batchLabel = batchLabel.cuda()
                batchIndex = batchIndex.cuda()

                # print(f"GPU {rank} got {len(batchData)} samples")

                batchPrediction = net(batchData)
                loss = lossFunction(batchPrediction, batchLabel)

                losses.Update(loss.item(), batchData.size(0))
                top1accuracy = TopKAccuracy(batchPrediction, batchLabel, topk=(1,))
                top5accuracy = TopKAccuracy(batchPrediction, batchLabel, topk=(5,))
                top1Accuracies.Update(top1accuracy[0], batchData.size(0))
                top5Accuracies.Update(top5accuracy[0], batchData.size(0))

                batchTime.Update((time.perf_counter_ns() - startTime) / 1e6)

                batchIndexes.append(batchIndex)
                batchPredictions.append(batchPrediction)

        return batchIndexes, batchPredictions

    """
    Testing/inferencing in DDP can be really tricky

    Before you go further, you may need to know basics about how DDP sync tensors among GPUs
    It can be found at: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    and: https://pytorch.org/docs/master/notes/ddp.html
    You also may want to check the implementation of AverageMeter

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Attention: the following scenarios are discussed when:
        1. Sampler: drop_last = True
        2. Dataloader: drop_last = False (use default)
        
        Example:
        sampler = torch.utils.data.distributed.DistributedSampler(..., drop_last = True)
        loader = torch.utils.data.DataLoader(..., sampler = sampler, drop_last = False)

        or

        sampler = torch.utils.data.distributed.DistributedSampler(..., drop_last = True)
        loader = torch.utils.data.DataLoader(..., sampler = sampler)

        You can skip setting the drop_last of the dataloader explicitly since its default value is False

        If you print the drop_last attribute of these two object, they will show:
        sampler.drop_last: True
        loader.drop_last: False

        I'm still not sure how this parameters work internally,
        but the following disscusion is empirical therefore it should be fine
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Suppose a DDP program is runnning on 2 GPUs
    # if the batch size is 5, len(dataset) = 30, then everything is fine. Log:
        GPU 1 got 5 samples
        GPU 0 got 5 samples
        GPU 1 got 5 samples
        GPU 0 got 5 samples
        GPU 1 got 5 samples
        GPU 0 got 5 samples

        len(testLoader.sampler): 15
        worldSize: 2
        len(testLoader.dataset): 30

    # if the batch size is 6, len(dataset) = 30,
    the first 24 samples works fine and the rest 6 samples can also divided by 2 (num of GPU)
    then each GPU will handle 3 samples. Log:
        GPU 1 got 6 samples
        GPU 0 got 6 samples
        GPU 1 got 6 samples
        GPU 1 got 3 samples
        GPU 0 got 6 samples
        GPU 0 got 3 samples

        len(testLoader.sampler): 15
        worldSize: 2
        len(testLoader.dataset): 30

    # if the batch size is 5, but len(dataset) = 31,
    the first 30 samples works fine, but the last sample cannot be divided by 2 (num of GPU)
    Then an auxiliary dataset with 1 sample will be created:
        GPU 1 got 5 samples
        GPU 0 got 5 samples
        GPU 0 got 5 samples
        GPU 0 got 5 samples
        GPU 1 got 5 samples
        GPU 1 got 5 samples

        len(testLoader.sampler): 15
        worldSize: 2
        len(testLoader.dataset): 31

        GPU 0: Constructing auxiliary dataset
        GPU 0 got 1 samples

    # if the batch size is 5, len(dataset) = 33,
    the first 30 samples works normally, the first two of the last three will be send to the two GPUs
    the last one will be the auxiliary dataset:
        GPU 1 got 5 samples
        GPU 0 got 5 samples
        GPU 1 got 5 samples
        GPU 1 got 5 samples
        GPU 1 got 1 samples
        GPU 0 got 5 samples
        GPU 0 got 5 samples
        GPU 0 got 1 samples

        len(testLoader.sampler): 16
        worldSize: 2
        len(testLoader.dataset): 33

        GPU 0: Constructing auxiliary dataset
        GPU 0 got 1 samples
    """

    net.eval()
    batchIndexes, batchPredictions = EvaluateLoop(testLoader)

    """
    Continue the example with batch size = 5 and 33 samples
    After running the EvaluateLoop above, the pattern of the two GPUs handling the data can be:
    00000 11111 00000 11111 00000 11111 0 1 -
    # It may be another pattern like: 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01
    # But it doesn't matter, the following analysis should be the same
    The "-" at last represent the sample left which need to be handled specially
    Also, there are two sets of loss and accuracy AverageMeter objects on the two GPUs:
    GPU0: loss, accuracy objects carrying results of the 16 samples (1~5, 11~15, 21~25, 31)
    GPU1: loss, accuracy objects carrying results of the 16 samples (6~10, 16~20, 26~30, 32)
    """

    batchTime.AllReduce()
    losses.AllReduce()
    top1Accuracies.AllReduce()
    top5Accuracies.AllReduce()

    indexes = []
    predictions = []
    for batchIndex, batchPrediction in zip(batchIndexes, batchPredictions):
        gatheredBatchIndex = [torch.zeros_like(batchIndex) for _ in range(worldSize)]
        torch.distributed.all_gather(gatheredBatchIndex, batchIndex, async_op=False)
        indexes.append(torch.cat(gatheredBatchIndex, dim=0))

        gatheredBatchPrediction = [
            torch.zeros_like(batchPrediction) for _ in range(worldSize)
        ]
        torch.distributed.all_gather(
            gatheredBatchPrediction, batchPrediction, async_op=False
        )
        predictions.append(torch.cat(gatheredBatchPrediction, dim=0))

    indexes = torch.cat(indexes, dim=0)
    predictions = torch.cat(predictions, dim=0)

    """
    Here, we first do the all reduce for the loss and accuracy objects
    In the AllReduce function, tensors from different processes should be summed and synced:
    GPU0: loss, accuracy objects carrying results of the 32 samples (1~32)
    GPU1: loss, accuracy objects carrying results of the 32 samples (1~32)
    """

    # print(f"len(testLoader.sampler): {len(testLoader.sampler)}")
    # print(f"worldSize: {worldSize}")
    # print(f"len(testLoader.dataset): {len(testLoader.dataset)}")
    if len(testLoader.sampler) * worldSize < len(testLoader.dataset):
        # print(f"GPU {rank}: Constructing auxiliary dataset")
        auxiliaryTestDataset = torch.utils.data.Subset(
            testLoader.dataset,
            range(len(testLoader.sampler) * worldSize, len(testLoader.dataset)),
        )
        auxiliaryTestLoader = torch.utils.data.DataLoader(
            auxiliaryTestDataset,
            batch_size=testLoader.batch_size,
            shuffle=False,
            num_workers=testLoader.num_workers,
            persistent_workers=testLoader.persistent_workers,
            pin_memory=True,
        )
        auxiliaryBatchIndexes, auxiliaryBatchPredictions = EvaluateLoop(
            auxiliaryTestLoader
        )

        indexes = torch.cat([indexes] + auxiliaryBatchIndexes, dim=0)
        predictions = torch.cat([predictions] + auxiliaryBatchPredictions, dim=0)

    predictions = predictions[indexes.argsort()]

    """
    Here's the special cases: the dataset cannot be evenly divided by the batch size and the number of GPUs
    In this case, we create an auxiliary dataset and loader for the left sample
    Naturally, one may want to run the auxiliary dataset on one of the GPUs(processes)
    But normally the auxiliary dataset is very small (0 < numOfSample < numOfGPU) which cost subtle time to run
    We run the auxiliary dataset on all GPUs(processes) to avoid any possible bad affect on synchronization
    After running the auxiliary dataset, the pattern of the AverageMeter objects on the two  GPUs can be:
    GPU0: loss, accuracy objects carrying results of the 33 samples (1~33)
    GPU1: loss, accuracy objects carrying results of the 33 samples (1~33)

    and the exact output of the earlier log should be:
    ...
    GPU 0: Constructing auxiliary dataset
    GPU 0 got 1 samples
    GPU 1: Constructing auxiliary dataset
    GPU 1 got 1 samples

    if you only run the auxiliary dataset on GPU 0, it will be:
    GPU0: loss, accuracy objects carrying results of the 33 samples (1~33)
    GPU1: loss, accuracy objects carrying results of the 32 samples (1~32)
    In this case, do not use the result come from GPU 1, which is also a really weird scenario
    """

    if hasattr(top1Accuracies.Average, "item"):
        top1Accuracies.Average = top1Accuracies.Average.item()
    if hasattr(top5Accuracies.Average, "item"):
        top5Accuracies.Average = top5Accuracies.Average.item()

    if rank == 0:
        toPrint = ""
        toPrint += f"{name}:{' ' if len(name) > 4 else '       '}[loss, accuray] -> [{losses.Average:.4e}, ({top1Accuracies.Average:.2f}%, {top5Accuracies.Average:.2f}%)], "
        toPrint += f"bacth time -> {batchTime.Average:.6f}ms"
        print(toPrint)

    return losses.Average, top1Accuracies.Average, predictions


def Evaluate(testLoader, net, lossFunction, name, rank, worldSize, mode):
    """
    Evaluate the network with given data loader and loss function

    The evaluation process in DDP can be very tricky
    therefore the evaluation functions of these two mode are separated for easier debugging

    Parameters
    ----------
    The other parameters are expained in the function EvaluateSingle and EvaluateMultiple

    mode : str, default = "multiple"
        multiple or single
    """
    if mode == "single":
        return EvaluateSingle(testLoader, net, lossFunction, name)
    if mode == "multiple":
        return EvaluateMultiple(testLoader, net, lossFunction, name, rank, worldSize)
    raise Exception(f'Bad parameter "mode": {mode}')
