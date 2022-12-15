import time

import numpy as np
import torch
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
#from tqdm import tqdm
import torchvision.transforms as transforms

MEANS = (0.4914, 0.4822, 0.4465)
STD_DEV = (0.2023, 0.1994, 0.2010)
training_transformations = transforms.Compose(
    [
        transforms.RandomCrop(
            size=(32, 32), padding=4),
        transforms.RandomHorizontalFlip(
            0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STD_DEV)
    ]
)

testing_transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STD_DEV)
    ]
)


def downloadData(ROOT):
    # Now we load training amd testing data
    # Note, this is different to the DataLoader step. That shall come after this

    global MEANS, STD_DEV, training_transformations, testing_transformations

    training_data = datasets.CIFAR10(
        ROOT,
        train=True,
        download=True,
        transform=training_transformations
    )

    testing_data = datasets.CIFAR10(
        ROOT,
        train=False,
        download=True,
        transform=testing_transformations
    )

    return training_data, testing_data


class Basics():

    def __init__(self, model, optimizer, schedule, criterion, training, testing, modelName, validationThreshold):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.criterion = criterion
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.trainingHistory = {
            'accuracy': [],
            'loss': [],
            'time': []
        }
        self.testingHistory = {
            'accuracy': [],
            'loss': [],
        }
        self.validationHistory = {
            'accuracy': [],
            'loss': [],
        }
        self.trainingData = training
        self.testingData = testing
        self.trainingDataLoader = None
        self.testingDataLoader = None
        self.modelName = modelName
        self.TTAMetric = None
        self.TTAThreshold = float(validationThreshold)

    def __calculateAccuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def __getTime(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _countParameters(self):
        total_params = 0
        for name, parameter in self.model.named_parameters():
            params = parameter.numel()
            total_params += params
        return total_params

    def _trainModel(self, epoch, batchSize):
        """
        Abstraction layer for training steps
        1. Make predictions
        2. Calculate loss, accuracy
        3. Propogate loss backwards and update weights
        4. Record statistics, i.e loss, accuracy and time per epoch
        """

        # To indicate model in training phase
        # Will also turn on dropout in case we use it
        self.model.train()

        epoch_loss = 0
        epoch_acc = 0

        # with self.trainingDataLoader as tqdmObject:

        startTime = time.time()

        self.trainingDataLoader = torch.utils.data.DataLoader(
            dataset=self.trainingData,
            batch_size=int(batchSize),
            shuffle=True,
            num_workers=2,
        )

        for (x, y) in self.trainingDataLoader:
            # tqdmObject.set_description(desc=f"Epoch {epoch+1}")
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Step 1
            y_pred = self.model(x)

            # Step 2
            loss = self.criterion(y_pred, y)
            acc = self.__calculateAccuracy(y_pred, y)

            # Step 3
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # self.schedule.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # tqdmObject.set_postfix(accuracy=epoch_acc/len(self.trainingDataLoader),
            #                        loss=epoch_loss/len(self.trainingDataLoader))
        endTime = time.time()

        trainingMinutes, trainingSeconds = self.__getTime(startTime, endTime)
        trainingSeconds += trainingMinutes*60

        return epoch_acc/len(self.trainingDataLoader), epoch_loss/len(self.trainingDataLoader), trainingSeconds

    def _evaluateModel(self):
        """
        Abstraction layer for validation steps
        1. Make predictions
        2. Calculate loss, accuracy
        3. Record statistics, i.e loss, accuracy and time per epoch
        """

        # To indicate model in training phase
        # Will also turn on dropout in case we use it
        self.model.eval()

        epoch_loss = 0
        epoch_acc = 0

        self.testingDataLoader = torch.utils.data.DataLoader(
            dataset=self.testingData,
            batch_size=200,
            shuffle=True,
            num_workers=2,
        )

        with torch.no_grad():
            for (x, y) in self.testingDataLoader:

                x = x.to(self.device)
                y = y.to(self.device)

                # Step 1
                y_pred = self.model(x)

                # Step 2
                loss = self.criterion(y_pred, y)
                acc = self.__calculateAccuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_acc/len(self.testingDataLoader), epoch_loss/len(self.testingDataLoader)

    def trainEpochs(self, epochs=10, plot_results=True, validate=False, batchSize=32, verbose=False, TTA=False):
        """
        Will build more control into it, for now keeping it limited to

        epochs: how many epochs to train for
        validate: do we even have a validation dataset
        plot_results: whether to plot training, testing and validation losses, accuracies
        """
        print(
            "_____________________________________________________________________________")
        print(self.modelName)
        TTAStartTime = time.time()
        for epoch in range(epochs):
            startTime = time.time()

            trainingAccuracy, trainingLoss, trainingSeconds = self._trainModel(
                epoch, batchSize)
            if validate:
                testingAccuracy, testingLoss = self._evaluateModel()
            else:
                testingAccuracy, testingLoss = 0, 0
            self.schedule.step(testingLoss)
            endTime = time.time()

            epochMinutes, epochSeconds = self.__getTime(startTime, endTime)
            epochSeconds += epochMinutes * 60

            if verbose:
                print("Epoch:%3.0f|TrainingLoss:%.2f|TrainingAccuracy:%.2f|EpochTime:%.2fs|TestingLoss:%.2f|TestingAccuracy:%.2f" % (
                    epoch+1, trainingLoss, trainingAccuracy*100, epochSeconds, testingLoss, testingAccuracy*100))

            self.trainingHistory["loss"].append(trainingLoss)
            self.trainingHistory["accuracy"].append(trainingAccuracy)
            self.trainingHistory["time"].append(trainingSeconds)

            if validate:
                self.testingHistory["loss"].append(testingLoss)
                self.testingHistory["accuracy"].append(testingAccuracy)

                if TTA and testingAccuracy*100 > self.TTAThreshold:
                    print(testingAccuracy, self.TTAThreshold)
                    TTAEndTime = time.time()
                    TTAMinutes, TTASeconds = self.__getTime(
                        TTAStartTime, TTAEndTime)
                    TTASeconds += TTAMinutes * 60
                    self.TTAMetric = TTASeconds

                    break

        if plot_results:

            X = np.arange(1, len(self.trainingHistory['loss'])+1)

            plt.figure()
            plt.plot(X, self.trainingHistory['loss'], label='train_loss')
            if validate:
                plt.plot(X, self.testingHistory['loss'], label='test_loss')
            plt.legend()
            plt.savefig("./outputs/" + self.modelName +
                        str(epoch) + "LossVsEpochs.jpg")

            plt.figure()
            plt.plot(
                X, self.trainingHistory['accuracy'], label='train_acc')
            if validate:
                plt.plot(
                    X, self.testingHistory['accuracy'], label='test_acc')
            plt.legend()
            plt.savefig("./outputs/" + self.modelName +
                        str(epoch) + "AccVsEpochs.jpg")
