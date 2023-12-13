import csv
import torch

from dataclasses import dataclass, field

def PrintParameters(parameters, detail = True):
    """
    Print the number of parameters in a torch.nn.Module and return the total count

    Parameters
    ----------
    parameters : torch.nn.Module.parameters()

    detail : bool, default True
    """
    count = 0
    for parameter in parameters:
        size = 1
        for s in parameter.size():
            size *= s
        if detail:
            print(f"{parameter.size()} -> {size}")
        count += size

    print(f"Total -> {count}\n")
    return count

@dataclass
class Result:
    """
    A simple dataclass to store and help saving the training result including only loss and accuracy
    """
    TrainLoss: list = field(default_factory = list)
    ValidationLoss: list = field(default_factory = list)
    TestLoss: list = field(default_factory = list)
    
    TrainAccuracy: list = field(default_factory = list)
    ValidationAccuracy: list = field(default_factory = list)
    TestAccuracy: list = field(default_factory = list)

    def Append(self, trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy):
        self.TrainLoss.append(trainLoss)
        self.TrainAccuracy.append(trainAccuracy)
        self.ValidationLoss.append(validationLoss)
        self.ValidationAccuracy.append(validationAccuracy)
        self.TestLoss.append(testLoss)
        self.TestAccuracy.append(testAccuracy)
        
    def Save(self, path):
        with open(path, mode = "w", newline = "") as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(["TrainLoss", "TrainAccuracy", "ValidationLoss", "ValidationAccuracy", "TestLoss", "TestAccuracy"])
            csvWriter.writerows(zip(self.TrainLoss, self.TrainAccuracy, self.ValidationLoss, self.ValidationAccuracy, self.TestLoss, self.TestAccuracy))

def TimingCuda(target):
    """
    A useful function to measure the time of a function on cuda device

    Parameters
    ----------
    target : function
        A function with no parameters to be measured
    """
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)

    start.record()
    target()
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.Name = name
        self.Reset()

    def Reset(self):
        self.Value = 0
        self.Average = 0
        self.Sum = 0
        self.Count = 0

    def Update(self, value, n = 1):
        self.Value = value
        self.Sum += value * n
        self.Count += n
        self.Average = self.Sum / self.Count

    def AllReduce(self):
        device = torch.device("cuda")
        total = torch.tensor([self.Sum, self.Count], dtype = torch.float32, device = device)
        torch.distributed.all_reduce(total, torch.distributed.ReduceOp.SUM, async_op = False)
        self.Sum, self.Count = total.tolist()
        self.Average = self.Sum / self.Count

def TopKAccuracy(output, target, topk = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class StandardOutputDuplicator:
    """
    Helper class for duplicating the standard output to multiple streams

    A typical usage is to duplicate the standard output to both a file and the console
    
    Example:
    logFile = open("Log.log" mode = "w")
    sys.stdout = StandardOutputDuplicator(sys.stdout, logFile)

    Note that, only duplicate the output in the main process in multiprocessing scenario
    """
    def __init__(self, *streams):
        self.Streams = streams

    def write(self, data):
        for stream in self.Streams:
            stream.write(data)

    def flush(self):
        pass

def MemoryUsage(rank):
    """Memory usage in MB"""
    torch.cuda.synchronize(rank)
    freeMemory, totalMemory = torch.cuda.mem_get_info(rank)
    return (totalMemory - freeMemory) / 1024 / 1024
