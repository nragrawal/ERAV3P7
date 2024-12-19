import torch
import torch.nn as nn
import torch.nn.functional as F


# Large model with 100K+ number of parameters but with right skeleton
# Target:
# - Right skeleton
# - Conv -> Transition Block -> Conv -> Output Block
# Result
# - total # parameters = 134,996
# - max training accuracy = 99.71
# - max test accuracy = 98.98
# Analysis : 
# - Overfitting.
# - max pooling is at rf = 7, rf = 5 is recommended.
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=28x28, output_size=26x26, rf_in=1, rf_out=3, jin=1, jout=1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=26x26, output_size=24x24, rf_in=3, rf_out=5, jin=1, jout=1

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=24x24, output_size=22x22, rf_in=5, rf_out=7, jin=1, jout=1

        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size=22x22, output_size=22x22, rf_in=7, rf_out=7, jin=1, jout=1

        self.pool1 = nn.MaxPool2d(2, 2) 
        # input_size=22x22, output_size=11x11, rf_in=7, rf_out=8, jin=1, jout=2

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=11x11, output_size=9x9, rf_in=8, rf_out=12, jin=2, jout=2

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=9x9, output_size=7x7, rf_in=12, rf_out=16, jin=2, jout=2

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=7x7, output_size=5x5, rf_in=16, rf_out=20, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # input_size=5x5, output_size=5x5, rf_in=20, rf_out=24, jin=2, jout=2

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5, 5), padding=0, bias=False),
        ) # input_size=5x5, output_size=1x1, rf_in=24, rf_out=32, jin=2, jout=2

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Preserve the skeleton, reduce number of parameters.
# Target:
# - Move max pooling to rf = 5
# - Reduce number of parameters
# - Preserve the skeleton

# Result:
# - total # parameters = 11,068
# - max training accuracy = 99.28
# - max test accuracy = 98.5

# Analysis:
# - Model is still overfitting but the gap is reduced.
# - Max training accuracy is still not above 99%
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=28x28, output_size=26x26, rf_in=1, rf_out=3, jin=1, jout=1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=26x26, output_size=24x24, rf_in=3, rf_out=5, jin=1, jout=1

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size=24x24, output_size=24x24, rf_in=5, rf_out=5, jin=1, jout=1
        
        self.pool1 = nn.MaxPool2d(2, 2) 
        # input_size=24x24, output_size=12x12, rf_in=5, rf_out=6, jin=1, jout=2

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=12x12, output_size=10x10, rf_in=6, rf_out=10, jin=2, jout=2
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=10x10, output_size=8x8, rf_in=10, rf_out=14, jin=2, jout=2
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=8x8, output_size=6x6, rf_in=14, rf_out=18, jin=2, jout=2
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # input_size=6x6, output_size=6x6, rf_in=18, rf_out=22, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(6, 6), padding=0, bias=False),
        ) # input_size=6x6, output_size=1x1, rf_in=22, rf_out=32, jin=2, jout=2

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Add GAP to reduce number of parameters even more and bring it under 8K

# Target:
# - model with # parameters under 8K
# - keep the skeleton same
# - add gap which will also help with reducing the # parameters

# Result:
# - total # parameters = 7,468
# - max training accuracy = 99.00
# - max test accuracy = 98.81

# Analysis:
# - Good model
# - Gap between training and test accuracy is reduced, model is not overfitting now.
# - Skeleton is preserved, parameter count is under 8K so now we can focus on increasing accuracy
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=28x28, output_size=26x26, rf_in=1, rf_out=3, jin=1, jout=1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=26x26, output_size=24x24, rf_in=3, rf_out=5, jin=1, jout=1

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size=24x24, output_size=24x24, rf_in=5, rf_out=5, jin=1, jout=1
        
        self.pool1 = nn.MaxPool2d(2, 2) 
        # input_size=24x24, output_size=12x12, rf_in=5, rf_out=6, jin=1, jout=2

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=12x12, output_size=10x10, rf_in=6, rf_out=10, jin=2, jout=2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=10x10, output_size=8x8, rf_in=10, rf_out=14, jin=2, jout=2

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=8x8, output_size=6x6, rf_in=14, rf_out=18, jin=2, jout=2

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        ) # input_size=6x6, output_size=6x6, rf_in=18, rf_out=22, jin=2, jout=2

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # input_size=6x6, output_size=1x1, rf_in=22, rf_out=32, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Increase Accuracy

# Target:
# - add bn, regularization ( dropout ) to increase accuracy

# Result:
# - total # parameters = 7,600
# - max training accuracy = 99.08
# - max test accuracy = 99.28

# Analysis:
# - Accuracy is increased but is getting plateaued at 99.25
# - need to add some modification to images to see if it helps address accuracy plateauing
# - model is starting to underfit
class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # input_size=28x28, output_size=26x26, rf_in=1, rf_out=3, jin=1, jout=1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size=26x26, output_size=24x24, rf_in=3, rf_out=5, jin=1, jout=1

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size=24x24, output_size=24x24, rf_in=5, rf_out=5, jin=1, jout=1
        
        self.pool1 = nn.MaxPool2d(2, 2) 
        # input_size=24x24, output_size=12x12, rf_in=5, rf_out=6, jin=1, jout=2

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # input_size=12x12, output_size=10x10, rf_in=6, rf_out=10, jin=2, jout=2
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size=10x10, output_size=8x8, rf_in=10, rf_out=14, jin=2, jout=2
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size=8x8, output_size=6x6, rf_in=14, rf_out=18, jin=2, jout=2
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # input_size=6x6, output_size=6x6, rf_in=18, rf_out=22, jin=2, jout=2

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # input_size=6x6, output_size=1x1, rf_in=22, rf_out=32, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Modify dataset

# Target:
# - Add rotation
# - Increase test accuracy

# Result:
# - total # parameters = 7,600
# - max training accuracy = 98.90
# - max test accuracy = 99.31

# Analysis
# - test accuracy reduced, train accuracy increased. Expected as we have made training harder.
# - accuracy is not going beyond 99.3
# - The model is underfitting.
class Model5(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# Remove dropout
#
# Target:
# - try to remove dropout as its effectiveness in CNNs might not be that great and accuracy is not increasing.
# - Reach higher accuracy
#
# Result:
# - total # parameters = 7,600
# - max training accuracy = 99.38
# - max test accuracy = 99.38
#
# Analysis:
# - Worked! Dropout was causing issues.
# - 99.30 is hit in the 10th epoch and after that accuracy is fluctuating, so we can try to reduce LR after the 10th epoch for less variation.
# - Accuracy is still not going above 99.4
class Model6(nn.Module):
    def __init__(self):
        dropout_value = 0
        super(Model6, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Model 6 : Add Learning Rate Scheduler
# Target:
#
# - Reduce LR after 10th Epoch
# - Reach higher accuracy.
#
# Result
# - total # parameters = 7,264
# - max training accuracy = 99.44
# - max test accuracy = 99.52
# - 99.4% accuracy in the 11th epoch.
# - Max accuracy peaked at 13th epoch with 99.52%
# - Once it hit 99.4, accuracy was consistently above 99.4
#
# Analysis:
# - Reducing LR once you reach closer to highest accuracy helps.
# - Reducing LR too soon can make the model require more epochs to reach peak accuracy
# - Also optimized parameter count after reaching peak accuracy to go even lower.
class Model7(nn.Module):
    def __init__(self):
        dropout_value = 0
        super(Model7, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
