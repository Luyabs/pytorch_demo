import cv2 as cv
import torch
import torchvision
from model import MyLeNet

if __name__ == '__main__':
    img1 = cv.imread("datasets/dog.png")
    img2 = cv.imread("datasets/plane.png")
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize([32, 32])
    ])
    img1, img2 = trans(img1), trans(img2)
    img1 = torch.reshape(img1, (1, 3, 32, 32))
    img2 = torch.reshape(img2, (1, 3, 32, 32))

    # 序号对应标签
    dict_class = torchvision.datasets.CIFAR10("./CIFAR10_dataset", download=True).classes

    model = MyLeNet()
    model.load_state_dict(torch.load("my_neural_network.pth"))

    model.eval()  # 测试
    with torch.no_grad():
        output1 = model(img1)
        output2 = model(img2)
    print(output1)
    print(dict_class[output1.argmax(1)])
    print(output2)
    print(dict_class[output2.argmax(1)])