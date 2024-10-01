# Introduction

This file contains some notes related to the miniproject titled "Fuel Consumption Estimation".

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
