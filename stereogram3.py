from imageio import imread, imsave
from itertools import islice, product
from scipy import ndimage
import numpy as np
import OpenEXR # non-Windows: pip install openexr; Windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr
import skimage.transform # pip install scikit-image

def assertEqual(a, b, threshold=1e-6, limit=3):
    a, b = np.broadcast_arrays(a, b)
    indices = np.where(np.abs(b - a) > threshold)
    if not indices[0].size:
        return

    examples = "\n".join(
        "a[{}]={} != {}".format(
            ", ".join(str(x) for x in index),
            a[index],
            b[index]
        ) for index in islice(zip(*indices), limit)
    ) + ("..." if len(indices[0]) > limit else "")
    raise AssertionError(
        "arrays with shape {} differ by more than {}\n{}".format(
            a.shape, threshold, examples
        )
    )

class Arange:
    def __init__(self):
        self.cache = np.arange(0)

    def __call__(self, stop):
        if stop > len(self.cache):
            self.cache = np.arange(stop)
            self.cache.setflags(write=False)

        return self.cache[:stop]

arange = Arange()

def multiGet(a, *indices):
    a = np.asarray(a)
    extraShape = a.shape[:a.ndim - len(indices)]
    extraIndices = tuple(
        arange(n).reshape(
            (n,) + (1,) * (len(extraShape) - i)
        ) for i, n in enumerate(extraShape)
    )
    return a[extraIndices + tuple(indices)]

assertEqual(
    multiGet(
        np.arange(16).reshape((2, 2, 2, 2)),
        [[0, 1], [0, 1]], [[0, 1], [1, 0]]
    ),
    [[[0, 3], [5, 6]], [[8, 11], [13, 14]]]
)

def searchsorted(a, v, **kwargs):
    a = np.asarray(a)
    _, v = np.broadcast_arrays(a[..., :1], v)
    out = np.empty_like(v, dtype=int)
    for i in product(
        [...],
        *(
            range(n) if n > 1 else [slice(None)] \
                for n in a[..., :1].shape
        ),
    ):
        out[i] = np.searchsorted(np.squeeze(a[i]), v[i], **kwargs)

    return out

assertEqual(
    searchsorted(
        [[[0, 1, 2]], [[1, 2, 3]]],
        [
            [[[0.5, 0.5], [1.5, 1.5]]],
            [[[2.5, 2.5], [1.5, 1.5]]]
        ]
    ),
    [
        [[[1, 1], [2, 2]], [[0, 0], [1, 1]]],
        [[[3, 3], [2, 2]], [[2, 2], [1, 1]]]
    ]
)

def useMap(abMap, a):
    a = np.clip(a, 0, abMap.shape[-1] - 1)
    ai = np.floor(a)
    np.minimum(ai, abMap.shape[-1] - 2, out=ai)
    aj = ai + 1
    bi = multiGet(abMap, ai.astype(int))
    bj = multiGet(abMap, aj.astype(int))
    return bi * (aj - a) + bj * (a - ai)

def unmap(abMap, b):
    b = np.clip(b, abMap[..., :1], abMap[..., -1:])
    aj = searchsorted(abMap, b, side="right")
    np.minimum(aj, abMap.shape[-1] - 1, out=aj)
    ai = aj - 1
    bi, bj = multiGet(abMap, ai), multiGet(abMap, aj)
    return (ai * (bj - b) + aj * (b - bi)) / (bj - bi)

testMap = np.array([0, 1, 3, 6])
assertEqual(
    useMap(testMap, [0.9, 1.9, 2.9]),
    [0.9, 2.8, 5.7]
)
assertEqual(
    unmap(testMap, [0.9, 2.8, 5.7]),
    [0.9, 1.9, 2.9]
)
assert unmap(np.array([0, 1, 1, 2]), 1) == 2
assertEqual(useMap(np.array([0, 1]), [-1, 3]), [0, 1])
assertEqual(unmap(np.array([0, 1]), [-1, 3]), [0, 1])
assertEqual(
    useMap(np.array([[0, 1], [2, 3]]), [[0.1], [0.9]]),
    [[0.1], [2.9]]
)
assertEqual(
    unmap(np.array([[0, 1], [2, 3]]), [[0.1], [2.9]]),
    [[0.1], [0.9]]
)

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
        image = np.moveaxis(channels[:3], 0, 2)
        image *= 0.8
        image **= 0.5
        imsave(str(path) + ".png", np.round(np.clip(image, 0, 1) * 255).astype(np.uint8))
    
    if channelNames[-1] == "Z":
        depthMap = channels[-1]
        imsave(str(path) + ".z.png", np.round((depthMap - np.max(depthMap)) / (np.min(depthMap) - np.max(depthMap)) * 255).astype(np.uint8))

    return channels

def adjustRange(a, old1, old2, new1, new2, out=None):
    factor = (new2 - new1) / (old2 - old1)
    out = np.multiply(a, factor, out=out)
    out += new1 - old1 * factor
    return out

def getGaussian(length, sigmasInFrame=3):
    x = np.linspace(-sigmasInFrame, sigmasInFrame, length, endpoint = True)
    np.square(x, out = x)
    x *= -0.5
    np.exp(x, out = x)
    x *= sigmasInFrame / (0.5 * length * np.sqrt(2 * np.pi))
    return x

testCase = 5

channels = readDepthFile("zmap{}.exr".format(testCase)).astype(float)
_, height, cWidth = channels.shape
# height //= 2
# cWidth //= 2
unit = np.sqrt(height * cWidth) / 10
print(unit)
channels = np.moveaxis(
    skimage.transform.resize(
        np.moveaxis(channels, 0, 2),
        (height, cWidth)
    ), 2, 0
)
cRadii = channels[-1]
adjustRange(cRadii, np.min(cRadii), np.max(cRadii), 0.726 * unit, 0.804 * unit, out=cRadii)

cStart = int(np.floor(np.min(cRadii[:, 0])))
width = cStart + cWidth - 1 + int(np.ceil(np.min(cRadii[:, -1])))

clMap = np.arange(cStart, cStart + cWidth) - cRadii
np.maximum.accumulate(clMap, axis=1, out=clMap) # pylint: disable=no-member
lcMap = unmap(clMap, np.arange(width))
del clMap
lRadii = useMap(cRadii, lcMap)
del lcMap

crMap = np.arange(cStart, cStart + cWidth) + cRadii
np.minimum.accumulate( # pylint: disable=no-member
    crMap[:, ::-1], axis=1, out=crMap[:, ::-1]
)
rcMap = unmap(crMap, np.arange(width))
del crMap
rRadii = useMap(cRadii, rcMap)
del rcMap

cImage = channels[:3]
np.clip(cImage, 0, 1, out=cImage)
cBlurred = np.empty_like(cImage)
for channel, blurredChannel in zip(cImage, cBlurred):
    ndimage.filters.gaussian_filter(
        channel, sigma=0.05 * unit, output=blurredChannel
    )
cImage -= cBlurred
adjustRange(cBlurred, 0, 1, 0.15, 0.8, out=cBlurred)

bcImage = np.zeros((cImage.shape[0], height, width))
bcMagnitudes = np.zeros((height, width))
adImage = np.zeros((cImage.shape[0], height, width))
adMagnitudes = np.zeros((height, width))

lrMap = np.arange(width) + 2 * lRadii
lcMap = np.arange(-cStart, width - cStart) + lRadii
lcMask = lcMap < cWidth - 1
bcImage += useMap(cBlurred, lcMap) * lcMask
bcMagnitudes += lcMask
radii = useMap(lRadii, lrMap)
lrMap += radii
lcMap = lrMap - cStart
del lrMap
lcMask = lcMap < cWidth - 1
adImage += useMap(cBlurred, lcMap) * lcMask
adMagnitudes += lcMask
del lcMap
del lcMask

rlMap = np.arange(width) - 2 * rRadii
rcMap = np.arange(-cStart, width - cStart) - rRadii
rcMask = rcMap >= 0
bcImage += useMap(cBlurred, rcMap) * rcMask
bcMagnitudes += rcMask
radii = useMap(rRadii, rlMap)
rlMap -= radii
rcMap = rlMap - cStart
del rlMap
rcMask = rcMap >= 0
adImage += useMap(cBlurred, rcMap) * rcMask
adMagnitudes += rcMask
del rcMap
del rcMask

bcImage /= bcMagnitudes
adImage /= adMagnitudes
blurred = 1.5 * bcImage - 0.5 * adImage
imsave("blurred{}.png".format(testCase), np.round(np.clip(np.moveaxis(blurred, 0, 2), 0, 1) * 255).astype(np.uint8))

cScores = np.mean(cImage, axis=0)
cScores *= cScores
cScores *= getGaussian(cWidth, sigmasInFrame=2.5)
cScores = ndimage.filters.gaussian_filter(cScores, sigma=0.13 * unit)
np.power(cScores, 10, out=cScores)

magnitudes = np.zeros((height, width))

iterations = 8
layerCount = 2 * iterations
merged = np.zeros((cImage.shape[0], height, width))

lrMap = np.broadcast_to(np.arange(width), (height, width)).astype(float)
for i in range(iterations):
    radii = useMap(lRadii, lrMap)
    lrMap += radii
    lcMap = lrMap - cStart
    lrMap += radii
    lcMask = lcMap < cWidth - 1
    weights = useMap(cScores, lcMap)
    weights *= lcMask
    merged += useMap(cImage, lcMap) * weights
    magnitudes += weights

rlMap = np.broadcast_to(np.arange(width), (height, width)).astype(float)
for i in range(iterations):
    radii = useMap(rRadii, rlMap)
    rlMap -= radii
    rcMap = rlMap - cStart
    rlMap -= radii
    rcMask = rcMap >= 0
    weights = useMap(cScores, rcMap)
    weights *= rcMask
    merged += useMap(cImage, rcMap) * weights
    magnitudes += weights

merged /= magnitudes
merged += blurred
imsave("gram{}.png".format(testCase), np.round(np.clip(np.moveaxis(merged, 0, 2), 0, 1) * 255).astype(np.uint8))
