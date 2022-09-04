import os
import math
from ai_train.training.supervised import Supervised
from ai_dataloader.dataset.omniglot import OmniglotDataloader
from ai_train.implementation.imageHandler import ImageHandler
import torch
import numpy as np
from PIL import Image
from torch.nn import functional

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = functional.pairwise_distance(output1, output2, keepdim = True).squeeze(1)

      loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                    (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

class CustomLoss(torch.nn.Module):
    def __init__(self, margin=0.0001):
        super(CustomLoss, self).__init__()
        self.__margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = functional.pairwise_distance(output1, output2, keepdim = True).squeeze(1)

        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                    (1 - label) * (1 / (torch.pow(euclidean_distance, 2) +self.__margin)))

        return loss

class SiameseOmniglot(Supervised):
    """
    Class to handle siamese neural network using omniglot
    """
    def __init__(self, path, scenario, model, desiredSizeImage=128):
        Supervised.__init__(self, path, scenario, model)
        super().__init__(path, scenario, model)

        self.initObjects(OmniglotDataloader(path))

        self.initialized = False
        self.targetWidth = desiredSizeImage
        self.targetHeight = self.targetWidth

        lossFileName = "loss.txt"
        lossFile = os.path.join(self.pathReports, lossFileName)
        accuracyName = "accuracy.txt"
        accuracyFile = os.path.join(self.pathReports, accuracyName)
        self.reportFiles.update({"lossFile" : lossFile})
        self.reportFiles.update({"accuracyFile" : accuracyFile})

    def initObjects(self, dataloaderClass):
        """
        Method to load dataloader class
        """
        self.__dataloader = dataloaderClass

    def normalization(self, image):
        """
        Method to normalize image
        """
        return image / 255

    def samplesToTorch(self, image1, image2, out):
        """
        Custom function to convert x and y to torch

        x : numpy array
        y : dict
        """
        image1 = torch.tensor(image1)
        image2 = torch.tensor(image2)
        out = torch.tensor(out)

        image1 = functional.interpolate(image1.unsqueeze(0).unsqueeze(0), size=(self.targetWidth, self.targetHeight)).squeeze(0).type(torch.DoubleTensor)
        image2 = functional.interpolate(image2.unsqueeze(0).unsqueeze(0), size=(self.targetWidth, self.targetHeight)).squeeze(0).type(torch.DoubleTensor)

        image1 = self.normalization(image1)
        image2 = self.normalization(image2)

        return image1, image2, out

    def __loadSample(self, index):
        """
        Tool to load a sample
        """
        sample = self.__dataloader.getSample(index)

        image = torch.tensor(sample[0])
        out = torch.tensor(sample[1])

        image = functional.interpolate(image.unsqueeze(0).unsqueeze(0), size=(self.targetWidth, self.targetHeight)).squeeze(0).type(torch.DoubleTensor)

        image = self.normalization(image)
        
        return image.unsqueeze(0), out

    def customDataloader(self, batchSize, iterations, train=True):
        """
        Custom implementation custom dataloader

        batch x channels x height x width
        """
        if train:
            dataset = "train"
        else:
            dataset = "test"

        if self.initialized:
            pass
        else:
            self.iteration = 0
            self.batchIndex = 0

        batchSampleOut = list()
        firstIteration = True

        samples = self.__dataloader.getRandomBatchSample(batchSize)
        for index in range(batchSize):
            sample = self.samplesToTorch(samples[0][index], samples[1][index], samples[2][index])

            if firstIteration:
                batchSampleImage1 = sample[0].unsqueeze(0)
                batchSampleImage2 = sample[1].unsqueeze(0)
                firstIteration = False
            else:
                batchSampleImage1 = torch.cat((batchSampleImage1, sample[0].unsqueeze(0)))
                batchSampleImage2 = torch.cat((batchSampleImage2, sample[1].unsqueeze(0)))
            batchSampleOut.append(sample[2])

        if iterations == self.iteration:
            finished = True
        else:
            finished = False

        outTensor = torch.tensor(batchSampleOut)

        self.iteration += 1

        return batchSampleImage1, batchSampleImage2, outTensor, finished

    def train(self, loss=CustomLoss(), optimizer="adam", learningRate=0.000001, iterations=10000, batchSize=16):
        """
        Method to perform training

        model : Model torch.nn.Module
        loss : Loss torch.nn.MSELoss for Siamese
        optimizer : String
        learningRate : Float
        numberEpochs : int
        batchSize : int
        """
        loss = loss
        model = self.loadModel(self.model)
        model.train()
        model = model.double()
        if optimizer == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=learningRate)

        while(True):
            optim.zero_grad()
            image1, image2, out, finished = self.customDataloader(batchSize, iterations)
            image1 = self.checkCuda(image1)
            image2 = self.checkCuda(image2)
            out = self.checkCuda(out)

            prediction1 = model(image1)
            prediction2 = model(image2)

            error = loss(prediction1, prediction2, out)

            error.backward()
            optim.step()

            self.saveModel(model)

            print("Loss : " + str(error))

            if finished:
                break

    def manualTestSiamese(self, indexImage1, indexImage2):
        """
        Method to manually test the siamese NN
        """
        model = self.loadModel(self.model)
        model.eval()
        model = model.double()

        image1, label1 = self.__loadSample(indexImage1)
        image2, label2 = self.__loadSample(indexImage2)

        image1 = self.checkCuda(image1)
        image2 = self.checkCuda(image2)

        prediction1 = model(image1)
        prediction2 = model(image2)

        euclidean_distance = functional.pairwise_distance(prediction1, prediction2, keepdim = True)

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

    def resizePrediction(self, prediction, shapeImage):
        """
        Method to resize prediction to original size

        prediction : torchTensor -> [left, right, top, bottom]
        shapeImage : Array -> channel, height, width

        return torchTensor
        """
        predictionList = []

        scaleFactorWidth = shapeImage[2]
        scaleFactorHeight = shapeImage[1]

        predictionList.append(prediction[0] * scaleFactorWidth)
        predictionList.append(prediction[1] * scaleFactorWidth)
        predictionList.append(prediction[2] * scaleFactorHeight)
        predictionList.append(prediction[3] * scaleFactorHeight)

        predictionResized = torch.tensor(predictionList)
        return predictionResized.type(torch.IntTensor)

    def normalizeInput(self, input):
        """
        Method to normalize input

        input : torchTensor -> (channel, heigth, width)

        return torchTensor
        """
        x = functional.interpolate(input.unsqueeze(0), size=(self.targetWidth, self.targetHeight)).type(torch.FloatTensor)

        normalization = self.__dataloader.getNormalizationParameters()
        return (x - normalization["average"]) / normalization["deviation"]

    def normalizeOutput(self, output, inputShape):
        """
        Method to normalize output

        output : List -> [left, right, top, bottom]

        return torchTensor
        """
        yList = []

        scaleFactorWidth = self.targetWidth / inputShape[2]
        scaleFactorHeight = self.targetHeight / inputShape[1]

        yList.append(output[0] * scaleFactorWidth)
        yList.append(output[1] * scaleFactorWidth)
        yList.append(output[2] * scaleFactorHeight)
        yList.append(output[3] * scaleFactorHeight)

        y = torch.tensor(yList)
        y = y / self.targetWidth

        return y