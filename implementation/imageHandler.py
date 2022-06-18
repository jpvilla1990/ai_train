from typing import List
from PIL import ImageDraw
import torch
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
        print(tensor.shape)
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
