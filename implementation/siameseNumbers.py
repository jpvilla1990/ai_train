import os
import math
import torch
from torch.nn import functional
from ai_train.implementation.siameseOmniglot import SiameseOmniglot
from ai_dataloader.dataset.numbers import NumbersDataloader

class SiameseNumbers(SiameseOmniglot):
    """
    Class to handle siamese neural network using omniglot
    """
    def __init__(self, path, scenario, model, desiredSizeImage=128):
        SiameseOmniglot.__init__(self, path, scenario, model, desiredSizeImage=desiredSizeImage)
        super().__init__(path, scenario, model, desiredSizeImage)
        self.initObjects(NumbersDataloader(path))
        self.__path = path

    def loadSupportVectors(self):
        """
        Method to load support vector

        return dict -> dict support vector torch
        """
        dataloader = NumbersDataloader(self.__path)
        supportVector = dataloader.loadSupportVector()
        firstIteration = True
        for key in list(supportVector.keys()):
            if firstIteration:
                batchSupportVector = self.checkCuda(self.processNumpyImage(supportVector[key])).unsqueeze(0)
                firstIteration = False
            else:
                batchSupportVector = torch.cat((batchSupportVector, self.checkCuda(self.processNumpyImage(supportVector[key])).unsqueeze(0)))

        return batchSupportVector

    def processNumpyImage(self, npImage):
        """
        Method to process numpy to torch image
        """
        image = torch.tensor(npImage)
        image = functional.interpolate(image.unsqueeze(0).unsqueeze(0), size=(self.targetWidth, self.targetHeight)).squeeze(0).type(torch.DoubleTensor)
        return self.normalization(image)

    def loadSampleFromPath(self, path):
        """
        Tool to load a sample
        """
        dataloader = NumbersDataloader(self.__path)
        sample = dataloader.getImage(path)

        image = self.checkCuda(self.processNumpyImage(sample))
        
        return image.unsqueeze(0)

    def classifyImageNumber(self, path):
        """
        Method to classify image by using siamese nn
        """
        model = self.loadModel(self.model)
        model.eval()
        model = model.double()

        image = self.loadSampleFromPath(path)

        supportVector = self.loadSupportVectors()
        supportPredictions = model(supportVector)

        prediction = model(image)

        firstIteration = True

        for index in range(len(supportPredictions)):
            euclideanDistance = functional.pairwise_distance(prediction, supportPredictions[index], keepdim = True)

            if firstIteration:
                minimumDistance = euclideanDistance
                minimumLabel = index
                firstIteration = False
            else:
                if euclideanDistance < minimumDistance:
                    minimumLabel = index
                    minimumDistance = euclideanDistance

        return minimumLabel

    def calculateTotalAccuracy(self):
        """
        Method to calculate total accuracy
        """
        dataloader = NumbersDataloader(self.__path)
        datasetPaths = dataloader.getDictPaths()
        correctCounter = 0
        counter = 0
        for key in list(datasetPaths.keys()):
            for imagePath in datasetPaths[key]:
                prediction = self.classifyImageNumber(imagePath)
                if prediction == key:
                    correctCounter += 1

                counter += 1
                print((correctCounter / counter) * 100)

        print(correctCounter / counter)
