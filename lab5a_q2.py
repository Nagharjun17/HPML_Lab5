from pprint import pprint

import numpy as np
import torch
import torch.nn.parallel as pll
import torch.optim as optim
from matplotlib import pyplot as plt
import sys

from resnet import *
from utils import Basics, downloadData


# Constants
ROOT = "./.data"
LEARNINGRATE = 0.01
MOMENTUM = 0.9
WEIGHTDECAY = 5e-4
EPOCHS = 2
setParams = {
    "params": None,
    "lr": LEARNINGRATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHTDECAY,
    "nesterov": True,
}

BASEPATH = "./outputs"

# Downloading data just once
trainingData, testingData = downloadData(ROOT)

# Using SGD as the optimizer
lossFunction = nn.CrossEntropyLoss()

modelHistories = {
    "trainTimes": [],
}

if __name__ == "__main__":

    try:
        numGpus = int(sys.argv[1])
    except:
        numGpus = 1

    print(f"Using {numGpus} gpus")

    # Every batch size is a new model
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = ResNet50().to(device)
    setParams["params"] = model.parameters()
    optimizer = optim.SGD(**setParams)
    lrSched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=0, factor=0.9)
    batchSize = 128

    try:
        dev = torch.cuda.get_device_name(device=None)
        currentTA = Basics(model, optimizer, lrSched, lossFunction,
                           trainingData, testingData, dev + "_BatchSize_" + str(batchSize) + "_GPUS_" + str(numGpus), 92)

        currentTA.trainEpochs(50, True, True, batchSize,
                              verbose=True, TTA=True)
        
        filepath = currentTA.modelName + "_" + \
            str(currentTA.TTAMetric) + "_" + torch.cuda.get_device_name(0)
        torch.save(model, filepath)

        # Sorry about this ugly looking statement
        # Since we only want to look at training times for the last epoch
        # and I'm tracking times for every epoch, this is how I get the time
        # for the second i.e in this case, last epoch
        modelHistories["trainTimes"].append(
            currentTA.trainingHistory["time"][-1])
        model = model.cpu()
        del model
        del optimizer
        torch.cuda.empty_cache()

        print(f"TTA 92 percentage is: {currentTA.TTAMetric} sec")

    except Exception as e:
        print(f"Error is: {e}")

