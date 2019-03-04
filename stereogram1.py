from imageio import imread, imsave
import numpy as np

def getLerpParams(x):
    lBound = np.floor(x).astype(int)
    uBound = lBound + 1
    phase = x - lBound
    return lBound, uBound, phase

def lerp(x, y, phase):
    return x * (1 - phase) + y * phase

def getRPhases(shifts, center):
    height, width = shifts.shape
    centerRadii = np.round(shifts[:, center] * 0.5).astype(int)
    rPhases = \
        np.arange(-center, width - center) / (2 * centerRadii[:, None]) + 0.5
    heightRange = np.arange(height)
    for x in range(center, width):
        xShifts = shifts[:, x]
        lBound, uBound, phase = getLerpParams(x - xShifts)
        rPhases[:, x] = lerp(
            rPhases[heightRange, lBound],
            rPhases[heightRange, uBound],
            phase
        ) + 1
    assert np.all(rPhases[heightRange, center - centerRadii] == 0)
    return rPhases

def getLPhases(shifts, center):
    rPhases = getRPhases(shifts[:, ::-1], shifts.shape[1] - 1 - center)
    return -rPhases[:, ::-1]

shifts = np.mean(imread("zmap1.png").astype(float), axis=2) * -4 / 255 + 62
height, zWidth = shifts.shape
zCenter = zWidth // 2
centerRadii = np.round(shifts[:, zCenter] * 0.5).astype(int)
rPhases = getRPhases(shifts, zCenter)
lPhases = getLPhases(shifts, zCenter)
minRadius = np.min(centerRadii)
pCenter = zCenter + minRadius
pWidth = zWidth + 2 * minRadius
phases = np.empty((height, pWidth))
for y in range(height):
    phases[y, :pCenter + 1] = lPhases[
        y,
        centerRadii[y] - minRadius:zCenter + centerRadii[y] + 1
    ]
    phases[y, pCenter:] = rPhases[
        y,
        zCenter - centerRadii[y]:zWidth + minRadius - centerRadii[y]
    ]
assert np.all(phases[:, pCenter] == 0)
pattern = imread("pattern1.png").astype(float)
patternHeight, patternWidth, _ = pattern.shape
lxIndices, uxIndices, xIndexPhases = getLerpParams(phases * patternWidth)
lxIndices = np.mod(lxIndices, patternWidth)
uxIndices = np.mod(uxIndices, patternWidth)
yIndices = np.mod(np.arange(height), patternHeight)[:, None]
lGram = pattern[yIndices, lxIndices]
rGram = pattern[yIndices, uxIndices]
gram = lerp(lGram, rGram, xIndexPhases[:, :, None])
imsave("gram1.png", np.round(gram).astype(np.uint8))
# imsave("phases1.png", np.round((phases - np.min(phases)) / (np.max(phases) - np.min(phases)) * 255).astype(np.uint8))
