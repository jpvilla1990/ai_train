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
        self.reports = "reports"

        self.__parametersName = "parameters.pth"
        self.__parametersNameBackUp = "parametersBackUp.pth"
        self.__counter = 0
        self.__backUpCounter = 100

        self.setScenario(path, scenario)

        self.reportFiles = dict()
        self.reportFiles.update(
            {
                "parametersFile" : self.__parametersFile,
                "parametersFileBackUp" : self.__parametersFileBackUp,
            }
        )

    def __checkCuda(self, tensor, target="cuda"):
        """
        Move tensor to cuda or cpu if available checking if cuda is available

        tensor : tensor
        target : String -> cuda, cpu
        """
        if torch.cuda.is_available():
            if target == "cpu":
                tensor.to("cpu")
            elif target == "cuda":
                tensor.to(device="cuda")
        return tensor

    def setBackUpCounter(self, backUpCounter):
        """
        Method to set the number of iterations to save in the back Up parameters
        """
        self.__backUpCounter = backUpCounter

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
        self.pathReports = os.path.join(self.__pathScenario, self.reports)

        if os.path.exists(self.__pathParameters) is False:
            os.makedirs(self.__pathParameters)
        if os.path.exists(self.pathReports) is False:
            os.makedirs(self.pathReports)

        self.__parametersFile = os.path.join(self.__pathParameters, self.__parametersName)
        self.__parametersFileBackUp = os.path.join(self.__parametersFolder, self.__parametersNameBackUp)

    def saveModel(self, model):
        """
        Method to save parameters from the model
        """
        model = self.__checkCuda(model, target="cpu")
        torch.save(model.state_dict(), self.__parametersFile)
        self.__counter += 1
        if self.__counter == self.__backUpCounter:
            torch.save(model.state_dict(), self.__parametersFileBackUp)
            self.__counter = 0

    def loadModel(self, model):
        """
        Method to load parameters to model if they exist, otherwise just let them as they are
        """
        if os.path.exists(self.__parametersFile):
            model.load_state_dict(torch.load(self.__parametersFile))
        else:
            if os.path.exists(self.__parametersFileBackUp):
                model.load_state_dict(torch.load(self.__parametersFileBackUp))

        model = self.__checkCuda(model)

        return model

    def train(self, model, loss, optimizer="adam", learningRate=0.0000001, numberEpochs=10, batchSize=16):
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
        model = self.loadModel(model)
        if optimizer == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=learningRate)

        while(True):
            optim.zero_grad()
            x, y, finished = self.customDataloader(batchSize, numberEpochs)
            x = self.__checkCuda(x)
            y = self.__checkCuda(y)

            prediction = model(x)

            error = loss(prediction, y)

            error.backward()
            optim.step()

            self.saveModel(model)
            self.reporting(y, prediction, error)

            print("Loss : " + str(error))

            if finished:
                break

    def resetTraining(self):
        """
        Method to reset training, erases the parameters and reporting files
        """
        for key in list(self.reportFiles.keys()):
            try:
                os.remove(self.reportFiles[key])
            except:
                pass

    def customDataloader(self, batchSize, epochs):
        """
        Overridable Public Method to implement dataloader

        batchSize : int

        return list of three elements -> tensors input, groundTruth with the batchSize included, boolean to indicate iterations are finished
        """
        raise Exception("customDataloader not Implemented yet, please implement it by overriding method customDataloader returning a tuple")

    def reporting(self, y, prediction, loss):
        """
        Overridable Public Method to implement reporting, intended to store in the folder 'reports' all data related to the training
        please use the public attribute self.pathReport to store the reports there, use variable self.reportFiles to add the files to be reported

        y : tensor Ground truth
        prediction : tensor prediction
        """
        raise Exception("reporting method not implemented yet, please implement it by overriding the method reporting using the public variable self.reports to store the data related to the training")