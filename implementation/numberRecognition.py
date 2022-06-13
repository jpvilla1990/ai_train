import math
from ai_train.training.supervised import Supervised
from ai_dataloader.dataset.streetViewHouseNumbers import SVHN
import torch

class NumberRecognition(Supervised):
    """
    Class that implements number recognition training
    """
    def __init__(self, path, scenario):
        super().__init__(path, scenario)
        self.__dataloader = SVHN(path)

        self.initialized = False

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

        yList.append(left)
        yList.append(right)
        yList.append(top)
        yList.append(bottom)

        y = torch.tensor(yList)

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
            batchSampleX.append(sample[0])
            batchSampleY.append(sample[1])

        if self.numberBatches - 1 == self.batchIndex:
            self.batchIndex = 0
            self.epoch += 1
        else:
            self.batchIndex += 1

        if numberEpochs == self.epoch:
            finished = True
        else:
            finished = False

        return batchSampleX, batchSampleY, finished
