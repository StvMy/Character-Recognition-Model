# Code agnostic code
import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"  
device

import gc
## TO CLEAR MEMORY
gc.collect() # Python's garbage collector
torch.cuda.empty_cache() # delete all caches and garbage

import requests
from pathlib import Path

# Download helper functions from learn PyTorch repo

if Path("helper_functions.py").is_file():
  print("functions already exist")
else:
  print("Downloading functions")
  functions = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open ("helper_functions.py", "wb") as f:
    f.write(functions.content)

from timeit import default_timer as timer

def print_train_time(start:float,
                     end:float,
                     device: torch.device = None):
  "Prints difference between start and end time."

  total_time = end - start
  print(f"Time on {device} : {total_time:.3f} seconds")
  return total_time

from helper_functions import accuracy_fn
from tqdm.auto import tqdm

def trainStep(model:torch.nn.Module,
              dataLoader:torch.utils.data.dataloader,
              loss_fn:torch.nn.Module,
              optimizer:torch.optim,
              device:torch.device = device):

  # Import tqdm for progress bar

  # set the seed
  torch.manual_seed(42)

  # Set model to cuda
  model.to(device)

  # Create training and test loop
  ### Training
  train_loss = 0
  train_acc = 0
#________________________________________________________________________________________________________
  # Add a loop to loop through the training batches
  for batch, (x,y) in tqdm(enumerate(dataLoader)):  ## x -> Features, y -> label
    x,y = x.to(device), y.to(device)

    model.train()

    optimizer.zero_grad()

    prediction = model(x)

    loss = loss_fn(prediction,y)
    train_loss += loss

    acc = accuracy_fn(y,torch.argmax(prediction,dim=1))
    train_acc += acc
    print(f"y:{y},\n pred:{torch.argmax(prediction,dim=1)}")

    loss.backward()

    optimizer.step()


    train_loss /= len(dataLoader)
    train_acc /= len(dataLoader)

    # print(y.dtype, y.min().item(), y.max().item())
    if batch % 400 == 0:
      print(f"looked at { (batch) * len(x) }/{len(dataLoader.dataset)} samples")
      print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")



  print("COMPLETE")


from google.colab import drive
drive.mount('/content/drive')      # THIS IS FOR GOOGLE COLAB USER

from torchvision import transforms
from PIL import Image
import torch
from pandas import DataFrame
from tqdm.auto import tqdm

from torch.utils.data import Dataset
class ImageLabelDataset(Dataset):                   ## same as convertingToTensor just without the transform and proceses
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

### Data Proccessing
import pandas as pd

## Load CSV
df =  pd.read_csv("/content/drive/My Drive/Dataset/hand writting recognition/english.csv") #Read csv
x,y = df["image"],df["label"] #split data into features and labels

## split into Train and Test
lenTrain = int(len(x)*80/100)
dataTrain = df[:lenTrain]
dataTest = df[lenTrain:]
wholeData = df

print(f"number of labels: {len(y.unique())}\nnumber of features: {len(x)}")
print(f"Data Train [:5] = {dataTrain[:5]}\nData Test [:5]= {dataTest[:5]}")

## Prepare all Data at once
import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
filePath = "/content/drive/My Drive/Dataset/hand writting recognition/Img/"
dataTensorRaw = wholeData["image"]
labelRaw = wholeData["label"]
labelRaw = encoder.fit_transform(labelRaw)  # convert all to 0..61 integers


transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((255,189)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

imageStack = []
labelStack = []
for idx, paths in enumerate(dataTensorRaw):
  start = timer()

  openImage = Image.open(filePath+paths).convert("L")
  imageStack.append(transform(openImage))
  
#---------------------------------------------------------------------------------------------------------------------------------#

  label = labelRaw[idx]
  labelStack.append(torch.tensor(label, dtype=torch.long).squeeze())

#------------------------------------------- V stack V -------------------------------------------------------------------------------#

  end = timer()
  print_train_time(start = start,end = end,device = ("cpu"))
  print(idx)

  gc.collect() # Python's garbage collector
  torch.cuda.empty_cache() # delete all caches and garbage


wholeDataTensor = ImageLabelDataset(imageStack,labelStack)
## Load Data
from torch.utils.data import DataLoader, Dataset
customWholeData = DataLoader(dataset= wholeDataTensor,
                              batch_size=32,
                              shuffle=True)


## MAKE MAP OF LABELS
indexLabels= []
for x, idxLabels in wholeDataTensor:
  indexLabels.append(idxLabels)

## Setup loss fn and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = cv_model.parameters(),
                            lr=0.0001)

## Dataset for train and test
epochs = 15
for epoch in tqdm(range(epochs)):
  trainStep(model=cv_model,
            dataLoader=customWholeData,               
            loss_fn=loss_fn,
            optimizer=optimizer)
  gc.collect() # Python's garbage collector
  torch.cuda.empty_cache() # delete all caches and garbage


## Save best perform

# Saving our PyTorch model
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "CNNmodel.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"saving path: {MODEL_SAVE_PATH}")
torch.save(cv_model.state_dict(),MODEL_SAVE_PATH,)
