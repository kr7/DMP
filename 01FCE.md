# Miniproject 1: Fuel Consumption Estimation

This file contains some notes related to the miniproject titled "Fuel Consumption Estimation".
You can execute the codes in this file in Google Colab ( https://colab.research.google.com/ ).

# Sprint 1: Business Understanding 

Definition of the particular goals of the project, understand the background

**Task [FCE1-01] - Decide about the details (e.g. type of fuel) of the estimation task**

- type of fuel: patrol
- country, region: Hungary
- type of car: small car
- scale: proof-of-concept

**Task [FCE1-02] - Identification of use cases: why is the estimation of fuel consumption important?**

- climate change, environmental effect
- fuel is limited
- save energy, money: accurate estimation of fuel consumption contributes to more accurate estimation of costs

**Task [FCE1-03] - Identification of potential customers / target?**

- car producers
- users of cars (e.g. taxi companies)
- government
- environmental activists

**Task [FCE1-04] - What influences fuel consumption?**

- driving style (speed, acceleration)
- weather, temperature
- tire pressure
- type of the road, whether shortest path is taken
- vehicle design (aerodynamics of the vehicle)
- traffic conditions (e.g. traffic jam)

**Task [FCE1-05] - Search for a dataset/database**

http://www.biointelligence.hu/mi/fuel_data_with_errors.txt

**Task [FCE1-06] - Decision about the infrastructure to analyse the data**

https://colab.research.google.com/

**Task [FCE1-07] - Learn about the tools / infrastructure**

```
print("Hello World!")
```

The "phylosophy" of for-loops in Python is different from that of for-loops in Java. In python, 
a for-loop iterates over the items of a list (or something similar, such as a range) which corresponds to the 
for-each-loop in Java. Which instruction belongs to the body of the for-loop is determined based on 
the indentation, therefore, indentation is ***essential*** in Python. Indentation is also important 
in case of "if"-s and function definitions.

```
for i in range(5,13):
  print("Hello World "+str(i))
  if i % 2 == 0:
    print(f"{i} is even")
  else:
    print(f"{i} is odd")

print("This does not belong to the body of the for-loop")
```

Definition of a function:

```
def square_of_a_number(x):
  return x**2
```

```
 square_of_a_number(5)
```

The same operator (+) may work differently in case of different types:

Both operands are integers:
```
x = 5+6
```

Both a1 and a2 are lists (concatenation):
```
a1 = [4,5,6]
a2 = [1,2,3]
a1+a2
```

Both a1 and a2 are numpy arrays (element-wise addition):
```
import numpy as np
a1 = np.array([4,5,6])
a2 = np.array([1,2,3])
a1+a2 
```

Python tutorial: https://www.w3schools.com/python/ 

**Task [FCE1-08] - Load the data**

```
import pandas as pd
data = pd.read_csv("http://www.biointelligence.hu/mi/fuel_data_with_errors.txt", sep="\t", header=0)
```

What is the result of these expressions?

```
data[10:20]
```

```
data["starttime"]
``` 

```
data["starttime"][10:20]
```

**Task [FCE1-09] - Calculation of basic statistics (exploratory data analysis)**

Calculation of the average value of the ***avg.cons.*** column:

```
sum_of_v = 0
for v in data["avg.cons."]:
  sum_of_v = sum_of_v + v
avg_of_v = sum_of_v / len(data["avg.cons."])
```

or

```
data["avg.cons."].mean()
```

Which of the above options do you prefer? 

Distinct values and their counts:

```
data.groupby("air conditioner").count()["date"]
```


# Sprint 2: Data Preprocessing

Import pandas and load the data:

```
import pandas as pd
data = pd.read_csv("http://www.biointelligence.hu/mi/fuel_data_with_errors.txt", sep="\t", header=0)
```

**Task [FCE2-01] Handle missing values**

Print the number of missing values for each column:

```
for column in list(data):
  number_of_missing_values = data[column].isnull().sum()
  print(f"{number_of_missing_values} {column}")
```

Print the number of different values for a column (in the example, for the "trafic" column):

```
data.groupby("trafic").count()["date"]
```

What to do with missing values? Replace by 
- zero,
- a random value (between the minimum and maximum of that column),
- average (mean),
- median?

The aforementioned basic statistics (min, max, mean, median...) can be calculated like this: 

```
data['starttemp'].min()
data['starttemp'].max()
data['starttemp'].mean()
data['starttemp'].median()
```

Replace missing values of "starttemp" by the median of the column and store the resulting dataset in a new dataframe:

```
data1 = data.copy()
data1.loc[data1['starttemp'].isnull(), 'starttemp'] = data1['starttemp'].median()
```

Check that there are no more missing values: 

```
data1['starttemp'].isnull().sum()
```

The missing values in other columns may be replaced by the medians of those columns in the same way.


But, is it a good idea to replace the missing values of "starttemp" by the median of the column?
Note that "starttemp" and "endtemp" highly correlate and they are never missing at the same time:

```
import matplotlib.pyplot as plt
valid_values = data['endtemp']<100
plt.scatter( data['starttemp'][valid_values], data['endtemp'][valid_values] );
```

Whenever "starttemp" is missing, we can replace it by the value of endtemp:

```
data2 = data.copy()
data2.loc[data2['starttemp'].isnull(), 'starttemp'] = data2['endtemp']
```

Similarly, we can replace the missing values of "endtemp" as well.

But the actual difference between starttemp and endtemp depends on the time.
In order to examine that, we will create a new column into which we extract the first two letters of "starttime",
and another column for the difference of the temperatures and we plot the median difference for each hour of the day:

```
data['hour'] = 0
for i in range(len(data)):
  data.loc[i, 'hour'] = data['starttime'][i][0:2]

data['temp_diff'] = data['endtemp']-data['starttemp']

data.groupby('hour')['temp_diff'].median().plot()
```

Finally, we can replace the missing values using that observed median difference:

```
median_difference_by_hour = data.groupby('hour')['temp_diff'].median()

for i in range(len(data)):
  if data['starttemp'].isnull()[i]:
    data.loc[i, 'starttemp'] = data['endtemp'][i]-median_difference_by_hour[ data['hour'][i] ]
```

We can replace the missing values of "endtemp" similarly.

**[FCE2-02] Deduplication**

In this miniproject, we will only elimate exact duplicates. (In this dataset, there are no approximate duplicates.)

```
def same_as_next(i):
  for column in list(data):
    if data[column][i] != data[column][i+1]:
      return False
  return True

keep = []
for i in range(len(data)-1):
  keep.append(not same_as_next(i))
keep.append(True)

data_dedup = pd.DataFrame(data[keep])
```

**[FCE2-03] Handling inconsistent values**

The column "air conditioner" has contains some inconsistent values: 

```
data.groupby("air conditioner").count()["date"]
```

We will correct them: 

```
data.loc[ data['air conditioner']=='offf', 'air conditioner' ] = 'off'
```

Other errors ("typos") of the same column and other columns may be corrected similarly. 


# Sprint 3: Visualization

Import necessary libraries:

```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```

Load the data:

```
data = pd.read_csv('http://www.biointelligence.hu/mi/fuel_data.txt', header=0, sep='\t')
```

Histogram:

```
data['starttemp'].hist();
```

```
data['starttemp'].hist( bins=5 );
```

```
data['starttemp'].hist( bins=( 5, 10, 15, 20, 25, 30) );
```

![histogram](https://github.com/user-attachments/assets/eda9db38-0624-44db-9078-2e54f670f2d3)


Bar chart and pie chart: 

```
ac_on = (data['air conditioner'] == "on").sum()
ac_off = (data['air conditioner'] == "off").sum()
plt.bar( [1,2], [ac_on, ac_off], tick_label=['AC on', 'AC off'], color=["b","g"]);
```

![bar_chart](https://github.com/user-attachments/assets/2427be38-92d0-47cd-8ede-405ddfe794f2)


```
plt.pie( [ac_on, ac_off], labels=['AC on','AC off']);
```

![pie_chart](https://github.com/user-attachments/assets/40c307c4-36a2-4f10-abf0-7dcc9ee5e9a5)


Percentiles and interquartile range of "starttemp" (and other columns):

```
p25 = np.percentile(data["starttemp"], 25)
p75 = np.percentile(data["starttemp"], 75)
interquartile_range = np.percentile(data["starttemp"], 75) - np.percentile(data["starttemp"], 25)
```

Boxplot: 

```
plt.boxplot(data['starttemp'], notch=True);
```

```
plt.boxplot( [ data[(data['fuel type']=='95FS') & (data['avg.cons.']<10)]['avg.cons.'],
               data[(data['fuel type']=='95+') & (data['avg.cons.']<10)]['avg.cons.']],
             labels = ['95FS', '95+'] );
```

![boxplot](https://github.com/user-attachments/assets/93c91419-25f8-4f7b-8f69-a0bd743ad3f3)



Scatter plot - simple example:

```
x = [1, 4, 10, 15]
y = [2, 3, 8, 9]
plt.scatter(x,y, marker='x')
```

More scatter plots:

```
plt.scatter( data[data['avg.cons.']<10]['speed'], data[data['avg.cons.']<10]['avg.cons.']);
```

```
relevant_data = pd.DataFrame(data[data['avg.cons.']<10])
relevant_data.reset_index()

colors = []
for tr in relevant_data['trafic']:
  if tr == 'low':
    colors.append('g')
  elif tr == 'normal':
    colors.append('b')
  else:
    colors.append('r')

plt.scatter( relevant_data['speed'], relevant_data['avg.cons.'], c = colors);
```

![scatterplot](https://github.com/user-attachments/assets/1348ee4e-e2d3-4c7f-92c9-3fb01f23115c)



```
data['temp'] = (data['starttemp']+data['endtemp'])/2

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = []
for i in range(len(data)):
  if data['fuel type'][i]=='95+':
    colors.append('r')
  else:
    colors.append('b')

for j in range(len(data)):
  if data['trafic'][j] == 'low':
    ax.scatter(data['speed'][j], data['avg.cons.'][j],
              data['temp'][j], c = colors[j], marker = 'o')
  elif data['trafic'][j] == 'normal':
    ax.scatter(data['speed'][j], data['avg.cons.'][j],
              data['temp'][j], c = colors[j], marker = 'x')
  else:
    ax.scatter(data['speed'][j], data['avg.cons.'][j],
              data['temp'][j], c = colors[j], marker = '^')
plt.show();
```

![scatterplot3d](https://github.com/user-attachments/assets/8da92e06-6bfc-4c85-907c-11222f98bb2e)

# Sprint 4: Training a neural network for fuel consumption estimation

Import necessary libraries: 

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
```

Load and select the data:

```
data = pd.read_csv('http://www.biointelligence.hu/mi/fuel_data.txt', header=0, sep='\t')

selected_data = pd.DataFrame()
selected_data['starttemp'] = data['starttemp']
selected_data['endtemp'] = data['endtemp']
selected_data['speed'] = data['speed']
train_data = np.array(selected_data)
train_labels = np.array( [ [y] for y in data['avg.cons.']])
```

**Please note: in order to have a statistically unbiased estimation of the quality (accuracy) of the neural network, we have to evaluate it on COMPLETELY NEW data that has never been used for anything during the process of training the network (including the selection of the best model if we train several models).** For simplicity, in this miniproject, we use all the data to train the network and we will NOT evaluate its accuracy.

Define a class that represents our neural network:

```
class ConsumptionNet(nn.Module):
  def __init__(self):
    super(ConsumptionNet, self).__init__()
    self.hidden = nn.Linear(3, 10)
    self.out = nn.Linear(10, 1)

  def forward(self, x):
    x = torch.relu(self.hidden(x))
    return self.out(x)
```

Training the neural network:

```
train_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(train_data), torch.Tensor(train_labels) )
trainloader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=1)

net = ConsumptionNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-5)

for epoch in range(1000):
  for inputs, targets in trainloader:
    optimizer.zero_grad()

    outputs = net(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

In this miniproject, our "app" is a simple HTML page. We want to "integrate" the neural network into this HTML page:

FCE.html:

```
<html>
<head>
<title>
  Fuel consumption estimation
</title>
</head>

<script type="text/javascript">
  function estimate_cons() {
    /* we will insert the code here that will reproduce the calculations that
       the neural network does */
    window.alert("Not implemented yet.")
  }
</script>

<body>
<h1>Fuel consumption estimation<h1>
<form name="input_data">
<table>
<tr>
  <td>Starttemp:</td>
  <td><input type="text" name="temp1" size=3></td>
</tr>
<tr>
  <td>Endtemp:</td>
  <td><input type="text" name="temp2" size=3></td>
</tr>
<tr>
  <td>Average speed:</td>
  <td><input type="text" name="speed" size=3></td>
</tr>
<tr>
  <td></td>
  <td><input type="button" onclick="estimate_cons()" value="Calculate"></td>
</tr>
</table>
</form>
</body>
</html>
```

In order to be able to reproduce the calcuations that the trained neural network does, we need to know that a neural network consists of **nodes** or **units** that are organized into layers and each unit performs
a weighted sum of its inputs. An *input of a unit* is either one of the inputs of the network or the outputs of another unit, called *activation*. Of course, we also need to know the actual weights of the trained neural network.
The following code can be used to print the weights of the trained neural network:

```
for p in net.parameters():
  print(p)
```

The output of the above code is something like this (but not exactly the same as the numbers below, because each time we train the network, the training starts from a different random initialization):

```
Parameter containing:
tensor([[ 0.2509, -0.2061,  0.1073],
        [-0.2059, -0.2937, -0.1009],
        [ 0.1150,  0.2625, -0.3610],
        [ 0.6465, -0.2293,  0.4714],
        [ 0.3817,  0.4118, -0.5667],
        [-0.3119, -0.4398,  0.0307],
        [ 0.0666,  0.1040,  0.1719],
        [ 0.3125,  0.1885, -0.3690],
        [ 0.1097,  0.7307, -0.4844],
        [-0.3408, -0.2709,  0.0361]], requires_grad=True)
Parameter containing:
tensor([-0.2061,  0.4432,  0.1330, -0.2794, -0.1213, -0.1389, -0.5187,  0.2188,
        -0.4756, -0.1425], requires_grad=True)
Parameter containing:
tensor([[-0.1407, -0.0554,  0.1907,  0.1611,  0.6042,  0.1442, -0.0315,  0.3453,
          0.6535, -0.2172]], requires_grad=True)
Parameter containing:
tensor([0.3992], requires_grad=True)
```

Our neural network has 

- an input layer containing 3 units,
- a hidden layer containing 10 units, and
- an output layer that contains a single unit.

The nodes in the input layer do not perform any computations, they just represent the input data, i.e., the values of (a) starttemp, (b) endtemp and (c) average speed. Each unit of the hidden layer calcuates a weighted sum of inputs and adds a number to this weighted sum. The weights used in the calculation of the weighted sum can be seen in the above tensors. Let us consider the first raw of the first tensor and the first value of the second tensor, that is:

```
Parameter containing:
tensor([[ 0.2509, -0.2061,  0.1073],
       ... )
Parameter containing:
tensor([-0.2061, ... ], requires_grad=True)
...
```

This means that the first hidden unit calculates the following weighted sum:

```
first hidden unit = 0.2509*starttemp + (-0.2061)*endtemp + 0.1073*speed + (-0.2061)
```

Because our activation function is ReLU, if the above weighted sum is negative, the activation of the hidden unit will be set to 0 (zero), otherwise the activation will be same as the above weighted sum.

The activations of the other hidden units may be calculated similarly. 

In particular, considering the weights in the first two tensors, the activations of the hidden units can be calculated by the following JavaScript code: 

```
    h1  =  0.2509*input1 - 0.2061*input2 + 0.1073*input3 - 0.2061;
    h2  = -0.2059*input1 - 0.2937*input2 - 0.1009*input3 + 0.4432;
    h3  =  0.1150*input1 + 0.2625*input2 - 0.3610*input3 + 0.1330;
    h4  =  0.6465*input1 - 0.2293*input2 + 0.4714*input3 - 0.2794;
    h5  =  0.3817*input1 + 0.4118*input2 - 0.5667*input3 - 0.1213;
    h6  = -0.3119*input1 - 0.4398*input2 + 0.0307*input3 - 0.1389;
    h7  =  0.0666*input1 + 0.1040*input2 + 0.1719*input3 - 0.5187;
    h8  =  0.3125*input1 + 0.1885*input2 - 0.3690*input3 + 0.2188;
    h9  =  0.1097*input1 + 0.7307*input2 - 0.4844*input3 - 0.4756;
    h10 = -0.3408*input1 - 0.2709*input2 + 0.0361*input3 - 0.1425;

    if (h1 < 0) { h1 = 0; }
    if (h2 < 0) { h2 = 0; }
    if (h3 < 0) { h3 = 0; }
    if (h4 < 0) { h4 = 0; }
    if (h5 < 0) { h5 = 0; }
    if (h6 < 0) { h6 = 0; }
    if (h7 < 0) { h7 = 0; }
    if (h8 < 0) { h8 = 0; }
    if (h9 < 0) { h9 = 0; }
    if (h10 < 0) { h10 = 0; }
```

Let us consider the last two tensors:

```
...
Parameter containing:
tensor([[-0.1407, -0.0554,  0.1907,  0.1611,  0.6042,  0.1442, -0.0315,  0.3453,
          0.6535, -0.2172]], requires_grad=True)
Parameter containing:
tensor([0.3992], requires_grad=True)
```

Based on these weights, the output of the network can be calculated as follows: 

```
out = -0.1407*h1 - 0.0554*h2 + 0.1907*h3 + 0.1611*h4 + 0.6042*h5 + 0.1442*h6 - 0.0315*h7 + 0.3453*h8 + 0.6535*h9 - 0.2172*h10 + 0.3992;
```

Putting all this together, our "app" with the code that reproduces the calculations of the neural network looks as follows: 

FCE.html:

```
<html>
<head>
<title>
  Fuel consumption estimation
</title>
</head>

<script type="text/javascript">
  function estimate_cons() {
    input1 = window.document.input_data.temp1.value;
    input2 = window.document.input_data.temp2.value;
    input3 = window.document.input_data.speed.value;

    h1  =  0.2509*input1 - 0.2061*input2 + 0.1073*input3 - 0.2061;
    h2  = -0.2059*input1 - 0.2937*input2 - 0.1009*input3 + 0.4432;
    h3  =  0.1150*input1 + 0.2625*input2 - 0.3610*input3 + 0.1330;
    h4  =  0.6465*input1 - 0.2293*input2 + 0.4714*input3 - 0.2794;
    h5  =  0.3817*input1 + 0.4118*input2 - 0.5667*input3 - 0.1213;
    h6  = -0.3119*input1 - 0.4398*input2 + 0.0307*input3 - 0.1389;
    h7  =  0.0666*input1 + 0.1040*input2 + 0.1719*input3 - 0.5187;
    h8  =  0.3125*input1 + 0.1885*input2 - 0.3690*input3 + 0.2188;
    h9  =  0.1097*input1 + 0.7307*input2 - 0.4844*input3 - 0.4756;
    h10 = -0.3408*input1 - 0.2709*input2 + 0.0361*input3 - 0.1425;

    if (h1 < 0) { h1 = 0; }
    if (h2 < 0) { h2 = 0; }
    if (h3 < 0) { h3 = 0; }
    if (h4 < 0) { h4 = 0; }
    if (h5 < 0) { h5 = 0; }
    if (h6 < 0) { h6 = 0; }
    if (h7 < 0) { h7 = 0; }
    if (h8 < 0) { h8 = 0; }
    if (h9 < 0) { h9 = 0; }
    if (h10 < 0) { h10 = 0; }

    out = -0.1407*h1 - 0.0554*h2 + 0.1907*h3 + 0.1611*h4 + 0.6042*h5 + 0.1442*h6 - 0.0315*h7 + 0.3453*h8 + 0.6535*h9 - 0.2172*h10 + 0.3992;

    window.alert("Becsült fogyasztás: "+out)
  }
</script>

<body>
<h1>Fuel consumption estimation<h1>
<form name="input_data">
<table>
<tr>
  <td>Starttemp:</td>
  <td><input type="text" name="temp1" size=3></td>
</tr>
<tr>
  <td>Endtemp:</td>
  <td><input type="text" name="temp2" size=3></td>
</tr>
<tr>
  <td>Average speed:</td>
  <td><input type="text" name="speed" size=3></td>
</tr>
<tr>
  <td></td>
  <td><input type="button" onclick="estimate_cons()" value="Calculate"></td>
</tr>
</table>
</form>
</body>
</html>
```

# Closing Remarks

In this example, we "translated" a neural network into a JavaScript code. Similarly, neural networks may be translated to the language of a specific (embedded) system, such as that of a car. The idea is as follows: 

- training a neural network may be performed conveniently on a plattform like Google Colab,
- once a neural network has been trained, the calculations performed by the trained neural network may be reproduced in the language of the application (the target system into which the neural network should be integrated).
