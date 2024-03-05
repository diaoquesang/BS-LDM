from config import config
from torchvision import transforms
import cv2 as cv


class myTransformMethod():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (config.image_size, config.image_size))  # 改变图像大小
        if img.shape[-1] == 3:  # HWC
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像


myTransform = {
    'trainTransform': transforms.Compose([
        myTransformMethod(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'testTransform': transforms.Compose([
        myTransformMethod(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}
