from typing import List
from PIL import ImageDraw, Image
import torch
import numpy as np
from torchvision import transforms

class ImageHandler(object):
    """
    Class to handle images with PIL and torch
    """
    def convertTensorToImage(tensor):
        """
        Static method to convert a tensor into a PIL image

        tensor : tensor of dimensions 3 x width x height
        """
        transform = transforms.ToPILImage()
        return transform(tensor)

    def addBoxToImage(image, boxDimensions):
        """
        Static method to add the box to the image

        image : PIL image
        boxDimensions : torch -> left, right, top, botton
        """
        points = list()

        boxDimensions.type(torch.IntTensor)

        points.append(int(boxDimensions[0]))
        points.append(int(boxDimensions[2]))
        points.append(int(boxDimensions[1]))
        points.append(int(boxDimensions[2]))
        points.append(int(boxDimensions[1]))
        points.append(int(boxDimensions[3]))
        points.append(int(boxDimensions[0]))
        points.append(int(boxDimensions[3]))

        imageDraw = ImageDraw.Draw(image)
        imageDraw.polygon(points)

        return image

    def saveImage(image, path, format="png"):
        """
        Static method to save PIL image in path

        image: PIL.Image
        path: String
        """
        image.save(path, format=format)

    def loadImage(path):
        """
        Static method to load PIL image from path

        path : String
        """
        return Image.open(path)

    def convertImageToTensor(image):
        """
        Static method to convert PIL image to torch tensor

        image : PIL

        return torchTensor (channels, heigth, width)
        """
        imageNp = np.asarray(image)
        return torch.tensor(imageNp).permute(2,0,1)



