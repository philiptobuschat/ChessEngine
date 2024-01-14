import torch
import torch.nn as nn

class Model_v2(nn.Module):
    # very large conv net
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
    # large conv net
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
    # conv net
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
    # conv net, broader than v4 but not deeper
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
    # fully connected
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
    
class Model_v7(nn.Module):
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v7, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=(12, 8, 8))
        
        self.relu = nn.ReLU()

        # Fully connected layers for regression
        self.fc1 = nn.Linear(636, 128)  # Adjust the input size based on your data shape
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.relu(out_3x3)
        out_5x5 = self.relu(out_5x5)
        out_8x8 = self.relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out
    
class Model_v8(nn.Module):
    # same as Model v7 but bigger
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v8, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*5, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*5, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*5, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(636*5, 128*5)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128*5, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))

        out = self.fc3(out)

        return out
    
class Model_v9(nn.Module):
    # same as Model v8 but even bigger
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v9, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(636*15, 128*15)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128*15, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))

        out = self.fc3(out)

        return out

class Model_v10(nn.Module):
    # same as Model v8 but deeper (not bigger)
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v10, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*5, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*5, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*5, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(636*5, 128*5)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128*5, 128*5)
        self.fc3 = nn.Linear(128*5, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.fc4(out)

        return out
    
class Model_v11(nn.Module):
    # same as Model v8 but deeper AND bigger
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v11, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(636*15, 128*15)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128*15, 128*15)
        self.fc3 = nn.Linear(128*15, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.fc4(out)

        return out

class Model_v12(nn.Module):
    # shallow and wide
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v12, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(636*15, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class Model_v13(nn.Module):
    # shallow and even wider
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v13, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*150, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*150, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*150, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(636*150, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class Model_v14(nn.Module):
    # shallow and even wider, but with pooling again
    # conv net, no padding
    def __init__(self):
        super(Model_v14, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool3d(kernel_size=(1, 4, 4))

        # Fully connected layers for regression
        self.fc1 = nn.Linear(540, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Apply Pooling
        out_3x3 = self.pool(out_3x3)
        out_5x5 = self.pool(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    
class Model_v15(nn.Module):
    # efficient conv layer, deep , but include direct input pooling
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v15, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*2, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*20, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*2 + 16*12*4 + 12*20 + 128, 128*8)
        self.fc2 = nn.Linear(128*8, 128*4)
        self.fc3 = nn.Linear(128*4, 128*2)
        self.fc4 = nn.Linear(128*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.fc4(out)

        return out
    
class Model_v16(nn.Module):
    # efficient conv layer, deep , but include direct input pooling (like v15, bit larger fc layers) 
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v16, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*2, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*20, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*2 + 16*12*4 + 12*20 + 128, 128*16)
        self.fc2 = nn.Linear(128*16, 128*16)
        self.fc3 = nn.Linear(128*16, 128*2)
        self.fc4 = nn.Linear(128*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.fc4(out)

        return out
    
class Model_v17(nn.Module):
    # same as v15 (input pooling) but more fc layers
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v17, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(9668, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.fc4(out)

        return out
    
class Model_v18(nn.Module):
    # same as Model v8 but deeper AND bigger (like 11 but streamlined)
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v18, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*10, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*15, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*10 + 16*12*15 + 12*15, 128*5)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128*5, 128*5)
        self.fc3 = nn.Linear(128*5, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.fc4(out)

        return out
    
class Model_v19(nn.Module):
    # same as Model v8 but a little bigger
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v19, self).__init__()

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*6, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*8, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*10, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*6 + 16*12*8 + 12*10, 128*5)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128*5, 128*5)
        self.fc3 = nn.Linear(128*5, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))

        out = self.fc4(out)

        return out
    
class Model_v20(nn.Module):
    # efficient conv layer, deep , but include direct input pooling
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v20, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*2, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*10, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*2 + 16*12*4 + 12*10 + 128, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc3(out))
        out = self.leaky_relu(self.fc4(out))
        out = self.fc5(out)

        return out
    
class Model_v21(nn.Module):
    # efficient conv layer, deep , but include direct input pooling and inputs themselves
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v21, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*2, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*10, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*2 + 16*12*4 + 12*10 + 128 + 12*8*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)
        input_flat = x.view(x.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool, input_flat), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc3(out))
        out = self.leaky_relu(self.fc4(out))
        out = self.fc5(out)

        return out

class Model_v22(nn.Module):
    # efficient conv layer, deep , but include direct input pooling and inputs themselves
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v22, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*2, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*10, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*2 + 16*12*4 + 12*10 + 128 + 12*8*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)
        input_flat = x.view(x.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool, input_flat), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc3(out))
        out = self.leaky_relu(self.fc4(out))
        out = self.fc5(out)

        return out
    
class Model_v23(nn.Module):
    # efficient conv layer, deep , but include direct input pooling and inputs themselves
    # conv net, no padding, no pooling
    def __init__(self):
        super(Model_v23, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*2, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*10, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*2 + 16*12*4 + 12*10 + 128 + 12*8*8, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)
        input_flat = x.view(x.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool, input_flat), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc3(out))
        out = self.leaky_relu(self.fc4(out))
        out = self.fc5(out)

        return out

class Model_v24(nn.Module):
    # based on models v8 - v23 build lean but strong model
    # direct input inclusion only via strong pooling
    # small number of convs -> efficient forward pass
    def __init__(self):
        super(Model_v24, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*3, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*6, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*8, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*3 + 16*12*6 + 12*8 + 128, 128*10)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128*10, 128*5)
        self.fc3 = nn.Linear(128*5, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.dropout(out)

        out = self.fc4(out)

        return out
    
class Model_v25(nn.Module):
    # based on models v8 - v23 build lean but strong model
    # direct input inclusion only via strong pooling
    # small number of convs -> efficient forward pass
    # v25 even leaner than v24
    def __init__(self):
        super(Model_v25, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*1, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*2, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*1 + 16*12*2 + 12*4 + 128, 128*6)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128*6, 128*3)
        self.fc3 = nn.Linear(128*3, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.dropout(out)

        out = self.fc4(out)

        return out

class Model_v26(nn.Module):
    # based on models v8 - v23 build lean but strong model
    # direct input inclusion only via strong pooling
    # small number of convs -> efficient forward pass
    # v26 bit less lean than v24
    def __init__(self):
        super(Model_v26, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*4, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*6, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*8, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*4 + 16*12*6 + 12*8 + 128, 128*12)  # Adjust the input size based on your data shape
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128*12, 128*6)
        self.fc3 = nn.Linear(128*6, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)

        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        out = self.dropout(out)

        out = self.fc4(out)

        return out
    
class Model_v27(nn.Module):
    # large model
    # conv net, no padding, no pooling, direct input pooling
    def __init__(self):
        super(Model_v27, self).__init__()

        # direct input pooling
        self.input_pool = nn.MaxPool3d(kernel_size=(6, 1, 1))

        # 3D convolutional layers with different kernel sizes
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels=12*20, kernel_size=(12, 3, 3))
        self.conv5x5 = nn.Conv3d(in_channels=1, out_channels=12*30, kernel_size=(12, 5, 5))
        self.conv8x8 = nn.Conv3d(in_channels=1, out_channels=12*50, kernel_size=(12, 8, 8))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(36*12*20 + 16*12*30 + 12*50 + 128, 1048)
        self.fc2 = nn.Linear(1048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        input_pool = self.input_pool(x)

        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_8x8 = self.conv8x8(x)

        # Apply ReLU activation to each branch
        out_3x3 = self.leaky_relu(out_3x3)
        out_5x5 = self.leaky_relu(out_5x5)
        out_8x8 = self.leaky_relu(out_8x8)

        # Flatten for fully connected layers
        out_3x3 = out_3x3.view(out_3x3.size(0), -1)
        out_5x5 = out_5x5.view(out_5x5.size(0), -1)
        out_8x8 = out_8x8.view(out_8x8.size(0), -1)
        input_pool = input_pool.view(input_pool.size(0), -1)

        # Concatenate the output from each conv branch
        out = torch.cat((out_3x3, out_5x5, out_8x8, input_pool), dim=1)
        out = self.dropout(out)

        # Fully connected layers for regression
        out = self.leaky_relu(self.fc1(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc3(out))
        out = self.leaky_relu(self.fc4(out))
        out = self.fc5(out)

        return out
    
class Model_v28(nn.Module):
    # large fully connected network
    def __init__(self):
        super(Model_v28, self).__init__()

        # Fully connected layers for regression
        self.fc1 = nn.Linear(12*8*8, 12*8*8 * 4)
        self.fc2 = nn.Linear(12*8*8 * 4, 12*8*8 * 1)
        self.fc3 = nn.Linear(12*8*8 * 1, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        x = x.view(x.size(0), -1)

        # Fully connected layers for regression
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        return out
    
    
class Model_v29(nn.Module):
    # very large fully connected network
    def __init__(self):
        super(Model_v29, self).__init__()

        # Fully connected layers for regression
        self.fc1 = nn.Linear(12*8*8, 12*8*8 * 8)
        self.fc2 = nn.Linear(12*8*8 * 8, 12*8*8 * 4)
        self.fc3 = nn.Linear(12*8*8 * 4, 12*8*8 * 1)
        self.fc4 = nn.Linear(12*8*8 * 1, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply 3D convolutions with different kernel sizes

        x = x.unsqueeze(1)
        x = x.view(x.size(0), -1)

        # Fully connected layers for regression
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return out