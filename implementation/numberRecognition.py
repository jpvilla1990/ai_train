import os
import math
from ai_train.training.supervised import Supervised
from ai_dataloader.dataset.streetViewHouseNumbers import SVHN
from ai_train.implementation.imageHandler import ImageHandler
import torch
import numpy as np
from PIL import Image
from torch.nn import functional

class NumberRecognition(Supervised):
    """
    Class that implements number recognition training
    """
    def __init__(self, path, scenario, model, desiredSizeImage=128):
        Supervised.__init__(self, path, scenario, model)
        super().__init__(path, scenario, model)

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

        return x, y

    def customDataloader(self, batchSize, numberEpochs, train=True):
        """
        Custom implementation custom dataloader

        batch x channels x height x width - left, right, top, bottom
        """
        if train:
            dataset = "train"
        else:
            dataset = "test"

        if self.initialized:
            pass
        else:
            self.epoch = 0
            self.batchIndex = 0
            self.sizeDataset = self.__dataloader.getDatasetSize(dataset)
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
            sample = self.__dataloader.getRandomSample(dataset)
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

    def predictBox(self, imagePath):
        """
        Method to predict box from image, saves png image with drawn box

        imagePath : String
        """
        image = ImageHandler.convertImageToTensor(
            ImageHandler.loadImage(imagePath)
        )

        x = self.normalizeInput(image)

        y = self.predict(x).squeeze(0)

        y = self.resizePrediction(y, image.shape)

        print(y)

        ImageHandler.saveImage(
            ImageHandler.addBoxToImage(ImageHandler.convertTensorToImage(image), y),
            "prediction.png",
            )

    def deactivateTrainableLayers(self, nLayers):
        """
        Method to deactivate trainable layers for fine tunning

        nLayers : int

        return Model
        """
        model = self.loadModel(self.model)
        n = 0
        for param in model.parameters():
            param.requires_grad = False
            n += 1

        counter = 0
        for param in model.parameters():
            if n - nLayers <= counter:
                param.requires_grad = True
            counter += 1

        return model


    def fineTune(self, imagesPath, coordinates, loss, optimizer="adam", learningRate=0.0000001, iterations=None, nLayers=2, threshold=0.8):
        """
        Method to fine tune last layers

        imagesPath : String -> pathImages
        coordinates : List -> [left, right, top, bottom]
        loss :  torch.nn.Loss loss function
        optimizer : String
        learningRate : Float
        iterations : int
        nLayers : int
        """
        images = []
        for i in range(len(imagesPath)):
            image = ImageHandler.convertImageToTensor(
                ImageHandler.loadImage(imagesPath[i])
            )

            xTensor = self.normalizeInput(image)
            yTensor = self.normalizeOutput(coordinates[i], image.shape)

            images.append(image)

            if i == 0:
                x = xTensor
                y = yTensor.unsqueeze(0)
            else:
                x = torch.cat([x, xTensor])
                y = torch.cat([y, yTensor.unsqueeze(0)])

        model = self.deactivateTrainableLayers(nLayers)

        model.train()
        if optimizer == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=learningRate)

        counter = 0
        while(True):
            optim.zero_grad()
            x = self.checkCuda(x)
            y = self.checkCuda(y)

            prediction = model(x)

            error = loss(prediction, y)
            accuracy = self.calculateAccuracy(y, prediction)

            error.backward()
            optim.step()

            self.saveModel(model)

            print("Loss : " + str(error))
            print("Accuracy : " + str(accuracy))

            if counter == iterations:
                break
            if accuracy > threshold:
                break
            counter += 1
