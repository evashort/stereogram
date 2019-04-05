import imageio
import numpy as np
import OpenEXR # non-Windows: pip install openexr; Windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr
from scipy import ndimage

def readDepthFile(path, channelNames="RGBZ"):
    depthFile = OpenEXR.InputFile(str(path))
    header = depthFile.header()
    for channelName in channelNames:
        channelHeader = header["channels"][channelName]
        assert channelHeader.type.v == 2 # float32
        assert (channelHeader.xSampling, channelHeader.ySampling) == (1, 1)

    viewBox = header["dataWindow"]
    width = viewBox.max.x - viewBox.min.x + 1
    height = viewBox.max.y - viewBox.min.y + 1

    channels = np.empty((len(channelNames), height, width))
    for i, channelName in enumerate(channelNames):
        buffer = depthFile.channel(channelName)
        assert len(buffer) == height * width * np.dtype(np.float32).itemsize
        channels[i] = np.frombuffer(buffer, dtype=np.float32).reshape(
            (height, width)
        )

    if channelNames[:3] == "RGB":
        image = channels[:3]
        image *= 0.8
        image **= 0.5

    return channels

testCase = 5

channels = readDepthFile("zmap{}.exr".format(testCase))
image = channels[:3]
blurred = np.empty_like(image)
for channel, blurredChannel in zip(image, blurred):
    ndimage.filters.gaussian_filter(channel, sigma=3, output=blurredChannel)

edges = image - blurred
grayEdges = np.mean(edges, axis=0)
scores = ndimage.filters.gaussian_filter(grayEdges * grayEdges, sigma=20)
np.power(scores, 10, out=scores)

shifts = 220 * np.arange(-3, 3) + 110
layers = np.empty((image.shape[0], len(shifts)) + image.shape[1:])
scoreLayers = np.empty((len(shifts),) + scores.shape)
for i, shift in enumerate(shifts):
    layers[:, i] = np.roll(edges, shift, axis=2)
    scoreLayers[i] = np.roll(scores, shift, axis=1)

scoreLayers /= np.sum(scoreLayers, axis=0)
merged = np.sum(layers * scoreLayers, axis=1)
blurred = np.roll(blurred, 110, axis=2) + np.roll(blurred, -110, axis=2)
blurred *= 0.5
result = blurred + merged

imageio.imsave("frequencytest{}.png".format(testCase), np.round(np.clip(np.moveaxis(result, 0, 2), 0, 1) * 255).astype(np.uint8))
imageio.imsave("blur{}.png".format(testCase), np.round(np.clip(np.moveaxis(blurred, 0, 2), 0, 1) * 255).astype(np.uint8))
#imageio.imsave("edges{}.png".format(testCase), np.round(np.minimum(np.abs(edges), 1) * 255).astype(np.uint8))
imageio.imsave("scores{}.png".format(testCase), np.round(scores / np.max(scores) * 255).astype(np.uint8))
imageio.imsave("merged{}.png".format(testCase), np.round(np.clip(np.abs(np.moveaxis(merged, 0, 2)), 0, 1) * 255).astype(np.uint8))

