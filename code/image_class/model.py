from torch import nn

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Useful parameters
        kernel = 3
        padding = kernel // 2

        self.cnn_layers = nn.Sequential(
          nn.Conv2d(3, 8, kernel, padding = padding),
          nn.MaxPool2d(2),
          nn.BatchNorm2d(8),
          nn.ReLU(),
          # H_out = W_out = (224 - 2)/2 + 1 = 224/2 = 112

          nn.Conv2d(8, 16, kernel, padding = padding),
          nn.MaxPool2d(2),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          # H_out = W_out =  112/2 = 56

          nn.Conv2d(16, 8, kernel, padding= padding),
          nn.MaxPool2d(2),
          nn.BatchNorm2d(8),
          nn.ReLU(),
          # H_out = W_out =  56/2 = 28

          nn.MaxPool2d(2)
          # H_out = W_out =  28/2 = 14
          #nn.Conv2d(16, 16, kernel, padding= padding),
          #nn.Upsample(scale_factor = 2),
          #nn.BatchNorm2d(16),
          #nn.ReLU()
          # H_out = W_out =  100*2 = 200

        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8*14**2, 132),
            nn.ReLU(),
            #nn.Linear(500, 50),
            #nn.ReLU(),
            nn.Dropout(),
            nn.Linear(132, 7)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x