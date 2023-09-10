import torch
import torch.nn as nn

class Model_v2(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Model_v2, self).__init__()
        # First Layer of Parallel Convolutional Layers
        self.conv1_2x2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=2, padding=1)
        self.conv2_4x4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=4, padding=2)
        self.conv3_6x6 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=6, padding=3)
        self.conv4_8x8 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=8, padding=4)
        
        # Second Layer of Parallel Convolutional Layers
        self.conv5_2x2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, padding=1)
        self.conv6_4x4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, padding=2)
        self.conv7_6x6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, padding=3)
        self.conv8_8x8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=8, padding=4)
        
        # Sequential Convolutional Layers
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1, bias=None)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with the specified dropout probability

    def forward(self, x):
        # First Layer of Parallel Convolutional Layers
        x1 = self.relu(self.conv1_2x2(x))
        x2 = self.relu(self.conv2_4x4(x))
        x3 = self.relu(self.conv3_6x6(x))
        x4 = self.relu(self.conv4_8x8(x))
        
        # Second Layer of Parallel Convolutional Layers
        x5 = self.pool(self.relu(self.conv5_2x2(x1)))
        x6 = self.pool(self.relu(self.conv6_4x4(x2)))
        x7 = self.pool(self.relu(self.conv7_6x6(x3)))
        x8 = self.pool(self.relu(self.conv8_8x8(x4)))
        
        # Concatenate the feature maps from parallel layers
        x_parallel = torch.cat((x5, x6, x7, x8), dim=1)

        # Sequential Convolutional Layers
        x9 = self.relu(self.conv9(x_parallel))
        x10 = self.relu(self.conv10(x9))
        x11 = self.pool(self.relu(self.conv11(x10)))
        
        x = x11.view(x11.size(0), -1)  # Flatten the feature maps
        x = self.dropout(self.relu(self.fc1(x)))  # Apply dropout after the first fully connected layer
        x = self.dropout(self.relu(self.fc2(x)))  # Apply dropout after the second fully connected layer
        x = self.dropout(self.relu(self.fc3(x)))  # Apply dropout after the third fully connected layer
        x = self.fc4(x)
        return x #torch.tanh(x)
    
class Model_v3(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Model_v3, self).__init__()
        # First Layer of Parallel Convolutional Layers
        self.conv1_2x2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=2, padding=1)
        self.conv2_4x4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=4, padding=2)
        self.conv3_6x6 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=6, padding=3)
        
        # Second Layer of Parallel Convolutional Layers
        self.conv4_2x2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, padding=1)
        self.conv5_4x4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, padding=2)
        self.conv6_6x6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, padding=3)
        
        # Sequential Convolutional Layers
        self.conv7 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1, bias=None)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with the specified dropout probability

    def forward(self, x):
        # First Layer of Parallel Convolutional Layers
        x1 = self.relu(self.conv1_2x2(x))
        x2 = self.relu(self.conv2_4x4(x))
        x3 = self.relu(self.conv3_6x6(x))
        
        # Second Layer of Parallel Convolutional Layers
        x4 = self.pool(self.relu(self.conv4_2x2(x1)))
        x5 = self.pool(self.relu(self.conv5_4x4(x2)))
        x6 = self.pool(self.relu(self.conv6_6x6(x3)))
        
        # Concatenate the feature maps from parallel layers
        x_parallel = torch.cat((x4, x5, x6), dim=1)

        # Sequential Convolutional Layers
        x7 = self.relu(self.conv7(x_parallel))
        x8 = self.pool(self.relu(self.conv8(x7)))
        
        x = x8.view(x8.size(0), -1)  # Flatten the feature maps
        x = self.dropout(self.relu(self.fc1(x)))  # Fully connected Layer
        x = self.dropout(self.relu(self.fc2(x)))  # Fully connected Layer
        x = self.fc3(x)
        return x
    
class Model_v4(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(Model_v4, self).__init__()
        # First Layer of Parallel Convolutional Layers
        self.conv1_2x2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=2, padding=1)
        self.conv2_4x4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=4, padding=2)
        self.conv3_6x6 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=6, padding=3)
        self.conv4_8x8 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=8, padding=4)
        
        # Second Layer of Parallel Convolutional Layers
        self.conv5_2x2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, padding=1)
        self.conv6_4x4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, padding=2)
        self.conv7_6x6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, padding=3)
        self.conv8_8x8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=8, padding=4)
        
        # Sequential Convolutional Layers
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1, bias=None)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with the specified dropout probability

    def forward(self, x):
        # First Layer of Parallel Convolutional Layers
        x1 = self.relu(self.conv1_2x2(x))
        x2 = self.relu(self.conv2_4x4(x))
        x3 = self.relu(self.conv3_6x6(x))
        x4 = self.relu(self.conv4_8x8(x))
        
        # Second Layer of Parallel Convolutional Layers
        x5 = self.pool(self.relu(self.conv5_2x2(x1)))
        x6 = self.pool(self.relu(self.conv6_4x4(x2)))
        x7 = self.pool(self.relu(self.conv7_6x6(x3)))
        x8 = self.pool(self.relu(self.conv8_8x8(x4)))
        
        # Concatenate the feature maps from parallel layers
        x_parallel = torch.cat((x5, x6, x7, x8), dim=1)

        # Sequential Convolutional Layers
        x9 = self.relu(self.conv9(x_parallel))
        x10 = self.relu(self.conv10(x9))
        x11 = self.pool(self.relu(self.conv11(x10)))
        
        x = x11.view(x11.size(0), -1)  # Flatten the feature maps
        x = self.dropout(self.relu(self.fc1(x)))  # Apply dropout after the first fully connected layer
        x = self.dropout(self.relu(self.fc2(x)))  # Apply dropout after the second fully connected layer
        x = self.dropout(self.relu(self.fc3(x)))  # Apply dropout after the third fully connected layer
        x = self.fc4(x)
        return x #torch.tanh(x)
    
class Model_v4(nn.Module):
    def __init__(self):
        super(Model_v4, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(12, 3, 3), padding=(0, 1, 1))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(12, 5, 5), padding=(0, 2, 2))
        self.conv7x7 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(12, 7, 7), padding=(0, 3, 3))
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(1, 4, 4))

        # Fully connected layers for regression
        self.fc1 = nn.Linear(3 * 12 * 1 * 2 * 2, 128)  # Adjust the input size based on your data shape
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_7x7 = self.conv7x7(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.relu(out_3x3)
        out_5x5 = self.relu(out_5x5)
        out_7x7 = self.relu(out_7x7)

        # Apply max pooling to each branch
        out_3x3 = self.pool(out_3x3)
        out_5x5 = self.pool(out_5x5)
        out_7x7 = self.pool(out_7x7)

        # Concatenate the output from each branch
        out = torch.cat((out_3x3, out_5x5, out_7x7), dim=1)
        out = self.dropout(out)

        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out
    
class Model_v5(nn.Module):
    def __init__(self):
        super(Model_v5, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=160, kernel_size=(12, 3, 3), padding=(0, 1, 1))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=160, kernel_size=(12, 5, 5), padding=(0, 2, 2))
        self.conv7x7 = nn.Conv3d(in_channels=1, out_channels=160, kernel_size=(12, 7, 7), padding=(0, 3, 3))
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(1, 4, 4))
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(160 * 3 * 1 * 2 * 2, 32)  # Adjust the input size based on your data shape

        self.fc2 = nn.Linear(32, 1)  # Adjust the input size based on your data shape


    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_7x7 = self.conv7x7(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.relu(out_3x3)
        out_5x5 = self.relu(out_5x5)
        out_7x7 = self.relu(out_7x7)

        # Apply max pooling to each branch
        out_3x3 = self.pool(out_3x3)
        out_5x5 = self.pool(out_5x5)
        out_7x7 = self.pool(out_7x7)

        # Concatenate the output from each branch
        out = torch.cat((out_3x3, out_5x5, out_7x7), dim=1)
        out = self.dropout(out)

        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)

        # Fully connected layers for regression
        out = self.relu(self.fc1(out))
        out = self.dropout(out)

        out = self.fc2(out)

        return out
    
class Model_v6(nn.Module):
    def __init__(self):
        super(Model_v6, self).__init__()

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12 * 8 * 8, 12 * 8)
        self.fc2 = nn.Linear(12 * 8, 12 * 8)
        self.fc3 = nn.Linear(12 * 8, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        # Fully connected layers for regression
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x