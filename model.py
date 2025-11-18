import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (7,7), stride = 2, padding = 3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d((3,3), padding = 1, stride = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, (3,3), padding = 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d((2,2), stride = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size = (1,1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size = (3,3), padding = 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size = (1,1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size = (3,3), padding = 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d((2,2), stride = 2),
        )
        
        conv4_block = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(512, 256, (1,1)),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(256, 512, (3,3), padding = 1),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            for _ in range(4)])
        
        self.conv4 = nn.Sequential(
            conv4_block,
            nn.Conv2d(512, 512, (1,1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, (3,3), padding = 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d((2,2), stride = 2)
        )

        self.conv5 = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(1024, 512, (1,1)),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(512, 1024, (3,3), padding = 1),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            for _ in range(2)])



        self.fc = nn.Sequential(
            nn.AvgPool2d((3, 3), padding = 1),
            nn.Flatten(start_dim = 1, end_dim = -1),
            nn.Linear(3 * 3 * 1024, 4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(2048, 200)
        )


    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return self.fc(x)
