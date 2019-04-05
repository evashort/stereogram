from imageio import imread, imsave
import numpy as np
import OpenEXR # non-Windows: pip install openexr; Windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr

def readOpenEXR(path):
    depthFile = OpenEXR.InputFile(str(path))
    header = depthFile.header()
    for channelName in "RGB":
        channel = header["channels"][channelName]
        assert channel.type.v == 2 # float32
        assert (channel.xSampling, channel.ySampling) == (1, 1)
    viewBox = header["dataWindow"]
    width = viewBox.max.x - viewBox.min.x + 1
    height = viewBox.max.y - viewBox.min.y + 1
    result = np.empty((height, width, 3))
    for i, channelName in enumerate("RGB"):
        buffer = depthFile.channel(channelName)
        assert len(buffer) == height * width * np.dtype(np.float32).itemsize
        result[..., i] = np.frombuffer(buffer, dtype=np.float32).reshape(
            (height, width)
        )
    
    result *= 0.8
    result **= 0.5
    imsave(str(path) + ".png", np.round(np.clip(result, 0, 1) * 255).astype(np.uint8))
    return result

im = readOpenEXR("zmap5.exr")
im *= 0.5
im += 0.25
shift = 118
a = im[:, :-3 * shift]
b = im[:, shift:-2 * shift]
c = im[:, 2 * shift:-shift]
d = im[:, 3 * shift:]
result = 0.75 * (b + c) - 0.25 * (a + d)
imsave("colortest5.png", np.round(np.clip(result, 0, 1) * 255).astype(np.uint8))
imsave("colorcontrol5.png", np.round(0.5 * (b + c) * 255).astype(np.uint8))
