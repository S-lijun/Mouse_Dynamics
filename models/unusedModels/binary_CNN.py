'''
This file defines the flexCNN model that is used in the project,
which is a flexible CNN model that allows you to specify the number
of convolutional and affine layers, as well as the kernel sizes.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


#-----------------------------------------------------------------------

'''
CNN model with 2 convolutional layers and 1 fully connected layer.
All convolutional layers have kernels of size 3. While we can just use
flexCNN, it can sometime be annoying with the model loading, so we've
created this backup just in case.
'''

class CNN_2convlayer_1afflayer(nn.Module):
    def __init__(self):
        super(CNN_2convlayer_1afflayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        
        return x

#-----------------------------------------------------------------------
'''
A flexible CNN class that allows you to specify the number of convolutional,
layers, affine layers, and the kernel sizes.

Right now, only supports up to 3 affine layers.

Input:
- conv_layers: number of convolutional layers
- aff_layers: number of affine layers
- kernel_sizes: list of kernel sizes for each convolutional layer
'''

class flexCNN(nn.Module):
    def __init__(self, conv_layers=1, aff_layers=1, kernel_sizes=None):

        super(flexCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.aff_layers = nn.ModuleList()

        # Default to kernel sizes of size 3
        if kernel_sizes is None:
            kernel_sizes = [3] * conv_layers
        else:
            conv_layers = len(kernel_sizes)

        # We start with 3 input channels (R,G,B)
        cur_channels = 3
        # The size of our image is 224x224
        image_size = 224

        for i in range(conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels=cur_channels, out_channels=32*(2**i), kernel_size=kernel_sizes[i]))
            cur_channels = 32 * (2**i)
            image_size -= (kernel_sizes[i] - 1)

        # Compute the length of the flattened tensor
        flat_length = cur_channels * (image_size ** 2)

        # if we have only one affine layer, immediately output predictions
        if aff_layers == 1:
            self.aff_layers.append(nn.Linear(flat_length, 1))
        elif aff_layers == 2:
            self.aff_layers.append(nn.Linear(flat_length, 128))
            self.aff_layers.append(nn.Linear(128, 1))
        elif aff_layers == 3:
            self.aff_layers.append(nn.Linear(flat_length, 256))
            self.aff_layers.append(nn.Linear(256, 128))
            self.aff_layers.append(nn.Linear(128, 1))

    def forward(self, x):
        # Convolutional layers with ReLU activations
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Flatten before fully connected layers
        x = x.view(x.size(0), -1)

        # Add ReLU activations in between each layer
        for i, fc in enumerate(self.aff_layers):
            x = fc(x)
            if i < len(self.aff_layers) - 1:
                x = F.relu(x)
            # No sigmoid here!
        return x

            
#-----------------------------------------------------------------------

'''
This is the CNN model used in the insider threat detection paper.

It is slightly modified to use sigmoid activation rather than softmax with 2 outputs.
'''

# Version 3
'''
class insiderThreatCNN(nn.Module):
    def __init__(self):
        super(insiderThreatCNN, self).__init__()

        # ----- Convolutional Layers -----
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Compute final flattened size -----
        # Input is 224x224
        # After Conv(3x3, stride=1) no padding: size -= 2
        # After MaxPool(2x2): size //= 2
        # Do this 4 times:
        # Conv1: 224 -> 222 -> Pool: 111
        # Conv2: 111 -> 109 -> Pool: 54
        # Conv3: 54 -> 52 -> Pool: 26
        # Conv4: 26 -> 24 -> Pool: 12
        # Final feature map: 128 x 12 x 12

        self.fc1 = nn.Linear(128 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # Binary output

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # [32, 111, 111]
        x = self.pool2(F.relu(self.conv2(x)))  # [64, 54, 54]
        x = self.pool3(F.relu(self.conv3(x)))  # [128, 26, 26]
        x = self.pool4(F.relu(self.conv4(x)))  # [128, 12, 12]

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.fc3(x))
        return x'''

#Version 1
class insiderThreatCNN(nn.Module):
    def __init__(self, input_size=224):
        super(insiderThreatCNN, self).__init__()
        self.input_size = input_size
        
        # Calculate the size after convolutions and pooling
        size_after_conv1 = input_size - 2  # 3x3 conv with no padding
        size_after_pool1 = size_after_conv1 // 2  # 2x2 maxpool
        size_after_conv2 = size_after_pool1 - 2
        size_after_pool2 = size_after_conv2 // 2
        size_after_conv3 = size_after_pool2 - 2
        size_after_pool3 = size_after_conv3 // 2
        size_after_conv4 = size_after_pool3 - 2
        size_after_pool4 = size_after_conv4 // 2
        
        # Final feature map size
        self.final_size = size_after_pool4
        self.fc_input_size = 128 * self.final_size * self.final_size
        #[Input: 3×224×224 RGB image]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3) # → output: [32, 222, 222]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # → output: [32, 111, 111]
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # → output: [64, 109, 109]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # → output: [64, 54, 54]
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)#  → output: [128, 52, 52]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)# → output: [128, 26, 26]
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)# → output: [128, 24, 24]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)# → output: [128, 12, 12]
        
        self.fc5 = nn.Linear(128 * 12 * 12, 1024)# → Dropout(p=0.5)
        self.fc6 = nn.Linear(1024, 512) # → Dropout(p=0.5)
        self.fc7 = nn.Linear(512, 1) # → Sigmoid

        # [Output: Scalar in (0,1)]
                # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # Convolutional layers followed by max pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        #print(f"[FORWARD] after conv1: {x.mean().item():.4f}, {x.std().item():.4f}")
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #print(f"[FORWARD] after conv2: {x.mean().item():.4f}, {x.std().item():.4f}")
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        #print(f"[FORWARD] after conv3: {x.mean().item():.4f}, {x.std().item():.4f}")
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        #print(f"[FORWARD] after conv4: {x.mean().item():.4f}, {x.std().item():.4f}")
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # → output: [18432]
        
        # Fully connected and dropout layers
        x = F.relu(self.fc5(x))
        #print(f"[FORWARD] after fc5: {x.mean().item():.4f}, {x.std().item():.4f}")
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc6(x))
        #print(f"[FORWARD] after fc6: {x.mean().item():.4f}, {x.std().item():.4f}")
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc7(x)
        #x = torch.sigmoid(x)
        #print(f"[FORWARD] final output: {x.mean().item():.4f}, {x.std().item():.4f}")
        return x

