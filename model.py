import torch
from torch import nn


class MyLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5),
                      stride=(1, 1), padding=2),  # stride默认为1 padding可以写same 免去计算(如果不相同 则按卷积层相关的计算公式去算padding)
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same'),  # padding=same维持形状不变
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),  # Flatten之后的线性层(展平后的输入大小64[channel]*4*4[shape]=1024)
            nn.ReLU(),
            nn.Linear(64, 10)  # Outputs之前的线性层(Hidden units64与Outputs10之间)
        )

    def forward(self, x):
        x = self.model1(x)
        # x = torch.softmax(x)
        return x


if __name__ == '__main__':
    model = MyLeNet()
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # GPU Train
    input = torch.ones(64, 3, 32, 32)
    output = model(input)
    print(output.shape)