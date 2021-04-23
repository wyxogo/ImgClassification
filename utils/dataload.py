import cv2
import numpy as np
class DataLoader(object):
    def __init__(self, path, label, relabel='false'):
        self.path = path
        self.label = label
        if relabel:
            self.label = [0,1,2,3,4,5,6,7,8,9]
    def data(self):
        x = []
        y = []
        for i in range(0,10):
            for f in range(0,100):
                l = str(100*i+f)
                # 读取图片数据
                images = cv2.imread("{}/{}.jpg".format(self.path,l))
                # 统一shape
                # image = cv2.resize(images, (256,256), interpolation=cv2.INTER_CUBIC)
                # 灰度直方图
                # hist = cv2.calcHist([images], [0, 1, 2], None, [256,256,256], [0.0, 256.0, 0.0, 256.0, 0.0, 256.0])
                hist = cv2.calcHist([images], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
                x.append(((hist / 255).flatten()))
                y.append(self.label[i])
        x = np.array(x)
        y = np.array(y)
        return x,y
