from darkflow.net.build import TFNet
from common.singleton import BaseClassSingleton
import imageProcessor as ip

_ERROR_PIXEL = 30

class DogDetector(BaseClassSingleton):
    def __init__(self):
        self.options = {
            'model': './cfg/yolo.cfg',
            'load': './bin/yolov2.weights',
            'threshold': 0.3,
            'gpu': 1.0
        }

        self.tfnet = TFNet(self.options)

    def detectsOneDog(self, img):
        result = self.tfnet.return_predict(img)
        for res in result:
            if res['label'] == 'dog':
                return res
        return False

    def getDogRect(self, result, originalImg):
        tl = result['topleft']
        br = result['bottomright']

        x = max(tl['x'] - round(_ERROR_PIXEL / 2), 0)
        y = max(tl['y'] - round(_ERROR_PIXEL / 2), 0)
        width = br['x'] - tl['x'] + _ERROR_PIXEL
        height = br['y'] - tl['y'] + _ERROR_PIXEL

        imgHeight, imgWidth = originalImg.shape[:2]

        if (x + width) >= imgWidth:
            width = imgWidth - x - 1
        if (y + height) >= imgHeight:
            height = imgHeight - y - 1
        return (x, y, width, height)