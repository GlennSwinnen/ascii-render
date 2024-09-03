import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from PIL import ImageFont, ImageDraw, Image


class PipelineStep():

    def process(self, image):
        return image, None
    
    def show_process(self, image, cmap="gray"):
        image, result = self.process(image)
        plt.imshow(image, cmap=cmap)
        plt.show()
        return image, result

class ResizeStep(PipelineStep):

    def __init__(self, size, interpolation=cv.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def process(self, image):
        image = cv.resize(image, self.size, self.interpolation)
        return image, None

class BilateralFilterStep(PipelineStep):

    def __init__(self, diameter, sigma_color, sigma_space):
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def process(self, image):
        image = cv.bilateralFilter(image, self.diameter, self.sigma_color, self.sigma_space)
        return image, None
    
class MinMaxNormalizeStep(PipelineStep):

    def process(self, image):
        image = (image - image.min()) / (image.max() - image.min())
        return image, None
    
class SobelFilterStep(PipelineStep):

    def __init__(self, ddepth, dx, dy, ksize):
        self.ddepth = ddepth
        self.dx = dx
        self.dy = dy
        self.ksize = ksize

    def process(self, image):
        image = cv.Sobel(image, self.ddepth, self.dx, self.dy, ksize=self.ksize)
        image = np.abs(image)
        return image, None
    
class AddWeightedStep(PipelineStep):

    def __init__(self, step1: PipelineStep, step2: PipelineStep, weight1=0.5, weight2=0.5):
        self.step1 = step1
        self.step2 = step2
        self.weight1 = weight1
        self.weight2 = weight2

    def process(self, image):
        image1, _ = self.step1.process(image)
        image2, _ = self.step2.process(image)
        image = cv.addWeighted(image1, self.weight1, image2, self.weight2, 0)
        return image, None

    def show_process(self, image, cmap="gray"):
        image1, _ = self.step1.show_process(image, cmap)
        image2, _ = self.step2.show_process(image, cmap)
        
        image = cv.addWeighted(image1, self.weight1, image2, self.weight2, 0)
        plt.imshow(image, cmap=cmap)
        plt.show()
        
        return image, None
    
class DilateStep(PipelineStep):

    def __init__(self, kernel, iterations):
        self.kernel = kernel
        self.iterations = iterations

    def process(self, image):
        image = cv.dilate(image, self.kernel, iterations=self.iterations)
        return image, None
    
class ThresholdStep(PipelineStep):

    def __init__(self, thresh, maxval, type=cv.THRESH_BINARY):
        self.thresh = thresh
        self.maxval = maxval
        self.type = type

    def process(self, image):
        _, image = cv.threshold(image, self.thresh, self.maxval, self.type)
        return image, None
    

class AsciiStep(PipelineStep):

    def __init__(self, palette, characters, superpixel_size, font_size, font_path):
        self.palette = palette
        self.characters = characters
        self.superpixel_size = superpixel_size
        self.font_size = font_size
        self.font_path = font_path

    def to_ascii(self, image):
        # find best match in each superpixel
        text = ""
        for y in range(0, image.shape[0], self.superpixel_size[0]):
            for x in range(0, image.shape[1], self.superpixel_size[1]):
                
                x2 = x + self.superpixel_size[1]
                if x2 > image.shape[1]:
                    break
                
                y2 = y + self.superpixel_size[0]
                if y2 > image.shape[0]:
                    break

                superpixel = image[y:y2, x:x2]
                matches = np.zeros((len(self.characters), 1, 1), dtype=np.float32)
                
                for i, ascii_image in enumerate(self.palette):
                    matches[i] = cv.matchTemplate(superpixel, ascii_image, cv.TM_SQDIFF)[0,0]
                
                arg = np.argmin(matches)
                text += self.characters[arg]

            text += "\n"
        return text
    
    def from_ascii(self, text, image_shape):
        font = ImageFont.truetype(self.font_path, self.font_size)
        image = Image.fromarray(np.zeros(image_shape))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), text, font=font)
        return np.array(image) * 255

    def process(self, image):
        text = self.to_ascii(image)
        image = self.from_ascii(text, image.shape)
        return image, text
    
    def show_process(self, image, cmap="gray"):
        text = self.to_ascii(image)
        print(text)
        
        image = self.from_ascii(text, image.shape)
        plt.imshow(image, cmap=cmap)
        plt.show()
        
        return image, text

class FileSource():

    def __init__(self, path, read_mode=cv.IMREAD_GRAYSCALE):
        self.path = path
        self.read_mode = read_mode

    def __call__(self):
        image = cv.imread(self.path, self.read_mode)
        image = image.astype(np.float32)
        yield image

class FileSink():

    def __init__(self, path):
        self.path = path

    def __call__(self, image, result):
        cv.imwrite(self.path, image, )
        
        with open(self.path + ".txt", "w") as file:
            file.write(result)

class Pipeline():

    def __init__(self, steps, image_source, image_sink):
        self.steps = steps
        self.image_source = image_source
        self.image_sink = image_sink

    def process(self):
        for image in self.image_source():
            for step in self.steps:
                image, result = step.process(image)
            self.image_sink(image, result)

    def show_process(self, cmap="gray"):
        for image in self.image_source():
            
            plt.imshow(image, cmap=cmap)
            plt.show()

            for step in self.steps:
                image, result = step.show_process(image, cmap=cmap)

            self.image_sink(image, result)
