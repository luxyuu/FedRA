import torch


class VGG16(torch.nn.Module):
    def __init__(self,In_channel=1,classes=4):
        super(VGG16, self).__init__()
        self.feature = torch.nn.Sequential(

            torch.nn.Conv1d(In_channel, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            
            torch.nn.Conv1d(16, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(32*6,64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    model = VGG16(In_channel=1,classes=4)
    input = torch.randn(size=(128,1,54))
    output = model(input)
    print(f"输出大小{output.shape}")
    # print(model)
