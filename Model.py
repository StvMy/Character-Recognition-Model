class model1(nn.Module):
    def __init__(self,in_shape, hidden_shape, out_shape):
        super().__init__()

        self.conv_block1 = nn.Sequential(
          nn.Conv2d(in_channels=in_shape,
                    out_channels=hidden_shape,
                    kernel_size=3,
                    stride=1,
                    padding=1),
          nn.LeakyReLU(),
          nn.Conv2d(in_channels=hidden_shape,
                    out_channels=hidden_shape,
                    kernel_size=3,
                    stride=1,
                    padding=1),
          nn.LeakyReLU(),
          nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block2 = nn.Sequential(
          nn.Conv2d(in_channels=hidden_shape,
                  out_channels=hidden_shape,
                  kernel_size=3,
                  stride=1,
                  padding=1),
          nn.LeakyReLU(),
          nn.Conv2d(in_channels=hidden_shape,
                  out_channels=hidden_shape,
                  kernel_size=3,
                  stride=1,
                  padding=1),

          nn.LeakyReLU(),
          nn.MaxPool2d(kernel_size=2)

        )

        self._to_linear = None  # find dynamically

        self.convs = nn.Sequential(
            self.conv_block1,
            self.conv_block2,)

        self._get_conv_output((1, 255, 189))  # input image size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = self._to_linear, out_features=out_shape))

    def _get_conv_output(self, shape):
        x = torch.randn(1, *shape)
        print(f"1 img shape : {x.shape}")

        x = self.convs(x)
        self._to_linear = x.view(1,-1).shape[1]

        print(f"1 img shape after conv: {x.shape}")
        print(f"dynamic shape: {self._to_linear}")


    def forward(self, data):
        print(data.shape)
        data = self.convs(data)
        print(data.shape)
        data = self.classifier(data)
        print(data.shape)
        return data

cv_model = model1(in_shape=1,
                  hidden_shape=300,
                  out_shape=62,
                  ).to(device)

