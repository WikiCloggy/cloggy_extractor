import numpy as np
import cv2
from skimage.morphology import skeletonize

_medianFilterKernalSize = 3
_medianFilterCount = 30

def deleteBackground(img, rect, mask=None):
    if mask is None:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    else:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    #return img*mask2[:, :, np.newaxis]
    return mask, mask2

def objectToWhite(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i][j] != 0:
                gray[i][j] = 255
    return gray

def drawRectangle(img, rect: tuple, color: tuple = (0, 255, 0), size: int = 7):
    newImg = img.copy()
    startXY = (rect[0], rect[1])
    endXY = (rect[0] + rect[2], rect[1] + rect[3])

    return cv2.rectangle(newImg, startXY, endXY, color, size)

def resizeImage(img, wanted_size, rect=None , maintain_ratio=False):
    if rect is None:
        x, y, width, height = (0, 0, img.shape[1], img.shape[0])
    else:
        x, y, width, height = rect
    data_width, data_height = wanted_size
    img = img[y:y + height, x:x + width]
    shape = list(img.shape)
    shape[0] = data_height
    shape[1] = data_width
    shape = tuple(shape)
    result = np.zeros(shape, dtype=img.dtype)

    if maintain_ratio:
        height_ratio = data_height / height
        resized_width = round(width * height_ratio)
        if resized_width < data_width:
            resized_img_size = (resized_width, data_height)
            img = cv2.resize(img, resized_img_size, 0, 0, cv2.INTER_LINEAR)
            width_space = round((data_width - resized_width) / 2)
            result[:data_height, width_space:width_space + resized_width] = img[:data_height, :resized_width]
        else:
            width_ratio = data_width / width
            resized_height = round(height * width_ratio)
            resized_img_size = (data_width, resized_height)
            img = cv2.resize(img, resized_img_size, 0, 0, cv2.INTER_LINEAR)
            height_space = round((data_height - resized_height) / 2)
            result[height_space:height_space + resized_height, :data_width] = img[:data_height, :data_width]
    else:
        result = cv2.resize(img, wanted_size, 0, 0, cv2.INTER_LINEAR)
    return result

def skeletonizer(img):
    skeleton = img
    skeleton.dtype = np.bool
    skeleton = skeletonize(img)
    skeleton.dtype = np.uint8
    return skeleton