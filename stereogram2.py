from imageio import imread, imsave
import numpy as np

testCase = 2

shifts = np.mean(imread("zmap{}.png".format(testCase)).astype(float), axis=2) * -4 / 255 + 62
height, width = shifts.shape
heightRange, widthRange = np.arange(height), np.arange(width)
positions = shifts + widthRange
center = 0.5 * (width - 1)
riftPosition = center + np.mean(shifts)
acceptableRiftIndices = np.where(
    positions >= riftPosition,
    widthRange,
    width
)
riftIndices = np.min(acceptableRiftIndices, axis=1) - 1
lPositions = positions[heightRange, riftIndices]
rPositions = positions[heightRange, riftIndices + 1]
riftOffsets = (riftPosition - lPositions) / (rPositions - lPositions)
assert np.all(riftOffsets > 0) and np.all(riftOffsets <= 1)
rift = riftIndices + riftOffsets
imsave("rift{}.png".format(testCase), np.round(255 * np.maximum(1 - np.abs(np.arange(width) - rift[:, None]), 0)).astype(np.uint8))

weights = np.array([
    [0, 1, 0, 0],
    [-0.5, 0, 0.5, 0],
    [1, -2.5, 2, -0.5],
    [-0.5, 1.5, -1.5, 0.5]
])
sampledWidth = width - weights.shape[1] + 1
sampledShifts = np.lib.stride_tricks.as_strided(
    shifts,
    shape=(height, sampledWidth, weights.shape[1]),
    strides=shifts.strides + shifts.strides[1:],
    writeable=False
)
powers = riftOffsets[:, None] ** np.arange(weights.shape[0])
alignedShifts = np.einsum(
    "...xs,ps,...p",
    sampledShifts,
    weights,
    powers
)

riftLeft, riftRight = np.min(riftIndices), np.max(riftIndices)
newWidth = sampledWidth + riftLeft - riftRight
newWidthRange = np.arange(newWidth)
xIndexMap = newWidthRange + (riftIndices - riftLeft)[:, None]
newShifts = alignedShifts[heightRange[:, None], xIndexMap]
imsave("newShifts{}.png".format(testCase), np.clip(np.round(255 * (newShifts - 62) / -4), 0, 255).astype(np.uint8))
