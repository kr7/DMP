# Miniproject: Recognition of Handwritten Digits

In this miniproject, we will train a neural network for the recognition of handwritten digits. 
The code can be executed in Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/) . 

## Import the necessary libraries

```
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from google.colab import widgets
```

## Load the data

```
data = np.loadtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data')
```

In order to understand how the data can be displayed, let us play a bit with the imshow function:

```
plt.imshow(np.array( [[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0]]) )
plt.show()
```

![image](https://github.com/user-attachments/assets/2e8b0f6a-a48c-4134-9c36-edd6e4945120)

Display the first image of the database:

```
image_size = (16,16)
an_image = np.reshape(data[0,0:256], image_size )
plt.imshow(an_image)
plt.show()
```

![image](https://github.com/user-attachments/assets/e60df876-9130-407f-93b5-37d944fdeb53)

Display widgets for a more interactive visualisation of the content of the database.

An image from each of the classes:

```
tb = widgets.TabBar([str(i) for i in range(10)], location='start')
for i in range(10):
  with tb.output_to(i):
    an_image = np.reshape(data[i*20,0:256], image_size )
    plt.imshow(an_image)
    plt.show()
```

10 images showing a '3':

```
tb = widgets.TabBar([str(i) for i in range(10)], location='start')
for i in range(10):
  with tb.output_to(i):
    an_image = np.reshape(data[60+i,0:256], image_size )
    plt.imshow(an_image)
    plt.show()
```

## Split the data into training and test data

```
train_data = data[:1093,0:256]
train_labels = data[:1093,256:266]
test_data = data[1093:,0:256]
test_labels = data[1093:,256:266]
```

## Transform the labels (one-hot-encoding -> simple numbers)

```
def ordinary_labels(raw_labels):
  o_lab = []
  for i in range(len(raw_labels)):
    o_lab.append( np.argmax(raw_labels[i,:]) )
  return np.array(o_lab)

train_labels = ordinary_labels(train_labels)
test_labels = ordinary_labels(test_labels)
```

## Define the neural network

```
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        number_of_units_in_the_first_hidden_layer = 100
        number_of_units_in_the_second_hidden_layer = 50

        self.fc1 = nn.Linear(256, 
                             number_of_units_in_the_first_hidden_layer)
        
        self.fc2 = nn.Linear(number_of_units_in_the_first_hidden_layer, 
                             number_of_units_in_the_second_hidden_layer)
        
        self.out = nn.Linear(number_of_units_in_the_second_hidden_layer, 
                             10) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
```

## Train the neural network

```
train_dataset = torch.utils.data.TensorDataset(
  torch.Tensor(train_data), torch.LongTensor(train_labels) )
trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=1)

net = DigitRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

running_loss = 0.0
running_n = 0

for epoch in range(100):  
  for inputs, targets in trainloader:
    optimizer.zero_grad()
    
    outputs = net(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    running_n = running_n + 1


  print('epoch %d, loss: %.3f' % (epoch + 1, running_loss / running_n))
  running_loss = 0.0
  running_n = 0   
```

## Evaluate the neural network on the test data

```
test_dataset = torch.utils.data.TensorDataset( 
      torch.Tensor(test_data), torch.LongTensor(test_labels)
)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

correct = 0
total = 0

with torch.no_grad():
  for inputs, targets in testloader:
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print("Correct: %d"%(correct))
```

## Training on GPU

In order to train our network on a GPU, we only need to add a few lines of code:

```
train_dataset = torch.utils.data.TensorDataset(
  torch.Tensor(train_data), torch.LongTensor(train_labels) )
trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=1)

net = DigitRecognizer()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

running_loss = 0.0
running_n = 0

for epoch in range(100):  
  for inputs, targets in trainloader:
    inputs = inputs.cuda()
    targets = targets.cuda()

    optimizer.zero_grad()

    outputs = net(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    running_n = running_n + 1

  print('epoch %d, loss: %.3f' % (epoch + 1, running_loss / running_n))
  running_loss = 0.0
  running_n = 0
```

## Evaluation on GPU

Similary, we can evaluate our neural network using the GPU:

```
test_dataset = torch.utils.data.TensorDataset( 
      torch.Tensor(test_data), torch.LongTensor(test_labels)
)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

correct = 0
total = 0

with torch.no_grad():
  for inputs, targets in testloader:
    inputs = inputs.cuda()
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu()
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print("Correct: %d"%(correct))
```
