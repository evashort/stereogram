from imageio import imread, imsave
from itertools import islice, product
import numpy as np
import OpenEXR # non-Windows: pip install openexr; Windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr

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
    ai = np.floor(a)
    aj = ai + 1
    bi = multiGet(abMap, ai.astype(int))
    bj = multiGet(abMap, aj.astype(int))
    return bi * (aj - a) + bj * (a - ai)

def useTiledMap(abMap, a):
    ai = np.floor(a)
    aj = ai + 1
    bi = multiGet(abMap, ai.astype(int) % abMap.shape[-1])
    bj = multiGet(abMap, aj.astype(int) % abMap.shape[-1])
    return bi * (aj - a) + bj * (a - ai)

def unmap(abMap, b):
    aj = searchsorted(abMap, b, side="right")
    assert np.all(aj > 0)
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
assertEqual(
    useMap([[0, 1], [2, 3]], [[0.1], [0.9]]),
    [[0.1], [2.9]]
)
assertEqual(
    unmap([[0, 1], [2, 3]], [[0.1], [2.9]]),
    [[0.1], [0.9]]
)

def isIncreasing(curve, testPoints):
    testIndices = np.floor(testPoints).astype(int)
    assert np.all(testIndices >= 0)
    return multiGet(curve, testIndices) < multiGet(curve, testIndices + 1)

assert np.all(
    isIncreasing([[0, 0, 1], [0, 1, 1]], [0.5, 1.5]) == \
        [[False, True], [True, False]]
)

def readDepthFile(path):
    depthFile = OpenEXR.InputFile(str(path))
    header = depthFile.header()
    zChannel = header["channels"]["Z"]
    assert zChannel.type.v == 2 # float32
    assert (zChannel.xSampling, zChannel.ySampling) == (1, 1)
    viewBox = header["dataWindow"]
    width = viewBox.max.x - viewBox.min.x + 1
    height = viewBox.max.y - viewBox.min.y + 1
    buffer = depthFile.channel("Z")
    assert len(buffer) == height * width * np.dtype(np.float32).itemsize
    depthMap = np.frombuffer(buffer, dtype=np.float32).reshape(
        (height, width)
    )
    imsave(str(path) + ".png", np.round((depthMap - np.max(depthMap)) / (np.min(depthMap) - np.max(depthMap)) * 255).astype(np.uint8))
    return depthMap

def adjustRange(a, old1, old2, new1, new2, out=None):
    factor = (new2 - new1) / (old2 - old1)
    out = np.multiply(a, factor, out=out)
    out += new1 - old1 * factor
    return out

testCase = 4

radii = readDepthFile("zmap{}.exr".format(testCase)).astype(float)
adjustRange(radii, np.min(radii), np.max(radii), 116, 124, out=radii)
height, cWidth = radii.shape

cOrigin = 0.5 * (cWidth - 1)
cxMap = np.arange(cWidth) - cOrigin

clMap = cxMap - radii
np.maximum.accumulate(clMap, axis=1, out=clMap) # pylint: disable=no-member
xStart = int(np.ceil(np.max(clMap[:, 0])))
xOrigin = -xStart

crMap = cxMap + radii
np.minimum.accumulate( # pylint: disable=no-member
    crMap[:, ::-1], axis=1, out=crMap[:, ::-1]
)
xStop = int(np.ceil(np.min(crMap[:, -1])))

xMap = np.broadcast_to(
    np.arange(xStart, xStop, dtype=float),
    (height, xStop - xStart)
).copy()
lxMap = xMap[:, :xOrigin]
rxMap = xMap[:, xOrigin + 1:]

expectedIterations = 0.5 * cWidth / (2 * np.mean(radii))

rIterations = 0
while True:
    rcMap = unmap(crMap, rxMap)
    rlMap = useMap(clMap, rcMap)
    mask = np.logical_and(rcMap > cOrigin, isIncreasing(clMap, rcMap))
    if not np.any(mask):
        break
    rxMap[mask] = rlMap[mask]
    rIterations += 1

print("{}/{}".format(rIterations, expectedIterations))

lIterations = 0
while True:
    lcMap = unmap(clMap, lxMap)
    lrMap = useMap(crMap, lcMap)
    mask = np.logical_and(lcMap < cOrigin, isIncreasing(crMap, lcMap))
    if not np.any(mask):
        break
    lxMap[mask] = lrMap[mask]
    lIterations += 1

print("{}/{}".format(lIterations, expectedIterations))

imsave("xMap{}.png".format(testCase), np.round((xMap - np.min(xMap)) / (np.max(xMap) - np.min(xMap)) * 255).astype(np.uint8))

pattern = imread("pattern2.jpg").astype(float)
pillar = pattern[np.arange(height) % pattern.shape[0]]
pillarChannels = np.moveaxis(pillar, 2, 0)
gramChannels = useTiledMap(pillarChannels, xMap + 0.5 * pillar.shape[1])
gram = np.moveaxis(gramChannels, 0, 2)
imsave("gram{}.png".format(testCase), np.round(gram).astype(np.uint8))
