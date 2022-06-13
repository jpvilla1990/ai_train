import os
import torch

class Supervised(object):
    """
    Class to handle supervised learning

    path : String -> Location where training workspace is located
    scenario : String -> sub folder in the workspace to store difference training instance
    """
    def __init__(self, path, scenario):

        self.__parametersFolder = "parameters"
        self.__statisticsFolder = "statistics"

        self.setScenario(path, scenario)

    def setScenario(self, path, scenario):
        """
        Method to set the scenario

        path : String
        scenario : String
        """
        self.__path = path
        self.__scenario = scenario

        self.__pathScenario = os.path.join(self.__path, self.__scenario)
        self.__pathParameters = os.path.join(self.__pathScenario, self.__parametersFolder)
        self.__pathStatistics = os.path.join(self.__pathScenario, self.__statisticsFolder)

        if os.path.exists(self.__pathParameters) is False:
            os.makedirs(self.__pathParameters)
        if os.path.exists(self.__pathStatistics) is False:
            os.makedirs(self.__pathStatistics)

    def train(self, model, loss, optimizer="adam", learningRate=0.001, numberEpochs=10, batchSize=16):
        """
        Method to perform training

        model : Model torch.nn.Module
        loss : Loss torch.nn.MSELoss
        optimizer : String
        learningRate : Float
        numberEpochs : int
        batchSize : int
        """
        loss = loss
        if optimizer == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=learningRate)

        while(True):
            optim.zero_grad()
            x, y, finished = self.customDataloader(batchSize, numberEpochs)
            print(len(x), len(y), finished)
            prediction = model(x)

            error = loss(prediction, y)

            loss.backward()
            optim.step()

            if finished:
                break

    def customDataloader(self, batchSize, epochs):
        """
        Overridable Public Method to implement dataloader

        batchSize : int

        return list of three elements -> tensors input, groundTruth with the batchSize included, boolean to indicate iterations are finished
        """
        raise Exception("CustomDataloader not Implemented yet, please implement it by overriding method customDataloader returning a tuple")