## MODEL

class model1(nn.Module):
    def __init__(self,in_shape, hidden_shape, out_shape):
        super().__init__()

        self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=in_shape,
                  out_channels=hidden_shape,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_shape,
                  out_channels=hidden_shape,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape,
                    out_channels=hidden_shape,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                    out_channels=hidden_shape,
                    kernel_size=3,
                    stride=1,
                    padding=1),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=758016,   ## Figure out how to make this calculation of hidden unit automatic
                    out_features=out_shape)
        )

    def forward(self, data):
        data = self.conv_block1(data)
        # print(f"1: {len(data.shape)}")
        data = self.conv_block2(data)
        # print(f"2: {len(data.shape)}")
        data = self.classifier(data)
        # print(f"3: {len(data.shape)}")

        return data

cv_model = model1(in_shape=1,
                  hidden_shape=256,
                  out_shape=62,
                  ).to(device)

