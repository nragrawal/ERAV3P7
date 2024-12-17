# MNIST Digit Classification with PyTorch

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification.

## Project Structure

project_folder/
├── model.py # CNN architecture definition
├── train.py # Training script
└── README.md # This file


## Requirements

- Python 3.11 (recommended)
- PyTorch
- torchvision
- tqdm
- numpy
- Pillow

## Installation

1. **Create and activate a virtual environment:**

```
bash
Create virtual environment
python3.11 -m venv venv
Activate virtual environment
On macOS/Linux:
source venv/bin/activate
On Windows:
.\venv\Scripts\activate

```

3. **Install dependencies:**

```
bash
Upgrade pip
pip install --upgrade pip
Install PyTorch ecosystem
pip install torch==2.1.0 torchvision==0.16.0
Install other required packages
pip install pillow==10.0.0
pip install numpy==1.24.3
pip install tqdm==4.65.0
```

3. **Verify the installation:**

```
bash
python -c "import torch; import torchvision; import PIL; import numpy; print('All imports successful!')"
```

## Usage

Run the training script:

```
bash
python train.py
```

The script will:
- Download the MNIST dataset automatically
- Train the model for 20 epochs
- Display progress bars for each epoch
- Show test accuracy after each epoch
- Display a summary table at the end

## Model Architecture

The CNN model (`model.py`) consists of:
- Input Block: 1→8 channels
- Convolution Block 1: 8→16 channels
- Transition Block: 16→10 channels + MaxPool
- Convolution Block 2: Multiple layers with varying channels
- Global Average Pooling
- Output: 10 classes (digits 0-9)

## Training Details

- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.01 with StepLR scheduler
- Batch Size: 128 (GPU) / 64 (CPU)
- Data Augmentation: Random rotation (-7° to 7°)

## Results

The script will display the following:
- Test accuracy after each epoch
- A summary table at the end with training and test losses, accuracies, and total time taken

