import os
import math
from ai_train.training.supervised import Supervised
from ai_dataloader.dataset.streetViewHouseNumbers import SVHN
from ai_train.implementation.imageHandler import ImageHandler
import torch
from torch.nn import functional

class NumberRecognition(Supervised):
    """
    Class that implements number recognition training
    """
    def __init__(self, path, scenario, desiredSizeImage=128):
        Supervised.__init__(self, path, scenario)
        super().__init__(path, scenario)

        self.initialized = False
        self.targetWidth = desiredSizeImage
        self.targetHeight = self.targetWidth

        self.__dataloader = SVHN(path)

        lossFileName = "loss.txt"
        lossFile = os.path.join(self.pathReports, lossFileName)
        accuracyName = "accuracy.txt"
        accuracyFile = os.path.join(self.pathReports, accuracyName)
        self.reportFiles.update({"lossFile" : lossFile})
        self.reportFiles.update({"accuracyFile" : accuracyFile})

    def normalization(self, x, y):
        """
        Method to normalize x, y

        x: tensor
        y: tensor
        """
        normalization = self.__dataloader.getNormalizationParameters()
        x = (x - normalization["average"]) / normalization["deviation"]
        y = y / self.targetWidth

        return x, y

    def samplesToTorch(self, x, y):
        """
        Custom function to convert x and y to torch

        x : numpy array
        y : dict
        """
        x = torch.tensor(x)
        x = x.permute(2,0,1)
        yList = []

        left = 0
        right = 0
        top = 0
        bottom = 0

        firstSample = True
        for key in list(y.keys()):
            if firstSample:
                left = y[key]["left"]
                right = left + y[key]["width"]
                top = y[key]["top"]
                bottom = top + y[key]["height"]
                firstSample = False
            else:
                newLeft = y[key]["left"]
                newRight = newLeft + y[key]["width"]
                newTop = y[key]["top"]
                newBottom = newTop + y[key]["height"]

                if newLeft < left:
                    left = newLeft
                if newRight > right:
                    right = newRight
                if newTop < top:
                    top = newTop
                if newBottom > bottom:
                    bottom = newBottom

        scaleFactorWidth = self.targetWidth / x.shape[2]
        scaleFactorHeight = self.targetHeight / x.shape[1]

        yList.append(left * scaleFactorWidth)
        yList.append(right * scaleFactorWidth)
        yList.append(top * scaleFactorHeight)
        yList.append(bottom * scaleFactorHeight)

        y = torch.tensor(yList)

        x = functional.interpolate(x.unsqueeze(0), size=(self.targetWidth, self.targetHeight)).squeeze(0).type(torch.DoubleTensor)

        x, y = self.normalization(x, y)

        #ImageHandler.saveImage(
        #    ImageHandler.addBoxToImage(ImageHandler.convertTensorToImage(x), y),
        #    "image.png",
        #    )

        return x, y

    def customDataloader(self, batchSize, numberEpochs):
        """
        Custom implementation custom dataloader

        batch x channels x height x width - left, right, top, bottom
        """
        if self.initialized:
            pass
        else:
            self.epoch = 0
            self.batchIndex = 0
            self.sizeDataset = self.__dataloader.getDatasetSize("train")
            self.numberBatches = math.floor(self.sizeDataset / batchSize)
            self.lastBatchSize = self.sizeDataset % batchSize
            self.extraLastBatch = False

            if self.lastBatchSize != 0:
                self.numberBatches += 1
                self.extraLastBatch = True

        if (self.extraLastBatch and self.numberBatches - 1 == self.batchIndex):
            batchSize = self.lastBatchSize

        batchSampleX = list()
        batchSampleY = list()
        for _ in range(batchSize):
            sample = self.__dataloader.getRandomSample("train")
            sample = self.samplesToTorch(sample[0], sample[1])
            batchSampleX.append(sample[0].unsqueeze(0))
            batchSampleY.append(sample[1].unsqueeze(0))

        if self.numberBatches - 1 == self.batchIndex:
            self.batchIndex = 0
            self.epoch += 1
        else:
            self.batchIndex += 1

        if numberEpochs == self.epoch:
            finished = True
        else:
            finished = False

        xTensor = torch.Tensor(batchSize, batchSampleX[0].shape[1], batchSampleX[0].shape[2], batchSampleX[0].shape[3])
        yTensor = torch.Tensor(batchSize, batchSampleY[0].shape[1])
        torch.cat(batchSampleX, out=xTensor)
        torch.cat(batchSampleY, out=yTensor)

        return xTensor, yTensor, finished

    def calculateAccuracy(self, y, prediction):
        """
        Method to calculate accuracy by calculating the intersection and dividing the union of the boxes
        """
        numberBatches = len(y)
        accuracySum = 0.0
        for index in range(numberBatches):
            leftY = y[index][0]
            rightY = y[index][1]
            topY = y[index][2]
            bottomY = y[index][3]

            leftPrediction = prediction[index][0]
            rightPrediction = prediction[index][1]
            topPrediction = prediction[index][2]
            bottomPrediction = prediction[index][3]

            if (rightY < leftPrediction or rightPrediction < leftY or bottomY < topPrediction or bottomPrediction < topY):
                accuracySum += 0.0
            else:

                horizontalCoordinates = [
                    float(leftY),
                    float(rightY),
                    float(leftPrediction),
                    float(rightPrediction),
                ]
                horizontalCoordinates.sort()

                verticalCoordinates = [
                    float(topY),
                    float(bottomY),
                    float(topPrediction),
                    float(bottomPrediction),
                ]
                verticalCoordinates.sort()

                horizontalDimensions = [
                    (horizontalCoordinates[1] - horizontalCoordinates[0]),
                    (horizontalCoordinates[2] - horizontalCoordinates[1]),
                    (horizontalCoordinates[3] - horizontalCoordinates[2]),
                ]

                verticalDimensions = [
                    (verticalCoordinates[1] - verticalCoordinates[0]),
                    (verticalCoordinates[2] - verticalCoordinates[1]),
                    (verticalCoordinates[3] - verticalCoordinates[2]),
                ]

                regionsOut = [
                    horizontalDimensions[0] * verticalDimensions[0],
                    horizontalDimensions[0] * verticalDimensions[1],
                    horizontalDimensions[1] * verticalDimensions[0],
                    horizontalDimensions[1] * verticalDimensions[2],
                    horizontalDimensions[2] * verticalDimensions[2],
                ]
                regionIn =  horizontalDimensions[1] * verticalDimensions[1]

                accuracySum += (regionIn) / (regionIn + sum(regionsOut))
        return accuracySum / numberBatches

    def saveLoss(self, loss):
        """
        Method to save the loss in a txt file

        loss : tensor
        """
        lossFile = open(self.reportFiles["lossFile"], "a+")
        lossFile.write(str(float(loss)) + "\n")
        lossFile.close()

    def saveAccuracy(self, accuracy):
        """
        Method to save the accuracy in a txt file
        """
        accuracyFile = open(self.reportFiles["accuracyFile"], "a+")
        accuracyFile.write(str(float(accuracy)) + "\n")
        accuracyFile.close()

    def reporting(self, y, prediction, loss):
        """
        Custom implementation of reporting
        """
        self.saveLoss(loss)
        accuracy = self.calculateAccuracy(y, prediction)
        print("Accuracy: " + str(accuracy))
        self.saveAccuracy(accuracy)

