from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class ConvNet(Module):
    def __init__(self, num_channels, classes, embedding_size, inference=False):
        '''
            # Inspired by LeNet
            
            Input shape should be (100,50)
            
            num_channels : Channels in the input image. (Black and white or RGB)
            inference: When true, the model returns one but last layer (embedding layer)
        '''
        super(ConvNet, self).__init__()

        self.inference = inference

        self.conv1 = Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv3 = Conv2d(in_channels=32, out_channels=50, kernel_size=(3, 3))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
  
        self.fc1 = Linear(in_features=2500, out_features=embedding_size)
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=embedding_size, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        
        if self.inference:
            return x

        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output