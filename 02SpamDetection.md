# Miniproject: Spam Detection

In this miniproject we will see, how we can use classifiers from the scikit-learn package. 
In particular, we will train a decision tree for spam classification. The code can be executed in Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/) . 

## Import the necessary libraries

```
import numpy as np
from sklearn.tree import DecisionTreeClassifier
```

## Load the data

Lines beginning with ! will be interpreted as operating system commands, therefore, we can download the data like this:

```
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
```

We can display the first few lines of the data like this: 

```
!head spambase.data
```

Load data into a numpy array:

```
delimiter = ","
data_with_labels = np.loadtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', delimiter=delimiter)
```

Display the size of the data, i.e., number of instances and number of columns:

```
np.shape(data_with_labels)
```


Split the data into training and test sets:

```
data = data_with_labels[:,:-1]
labels = data_with_labels[:,-1]

test_indices = np.array(range(int(len(data)/5)))*5+4
train_indices = [i for i in range(len(data)) if i not in test_indices]

data_train   = data[train_indices]
data_test    = data[test_indices]
labels_train = labels[train_indices]
labels_test  = labels[test_indices]
```

Download the file containing the column names:

```
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names
```

Display the first few lines of that file so that we can see at which line the column names start:

```
!head -n 35 spambase.names
```

Load the column names into an array (it will be useful to display the decision tree later):

```
column_names = []
with open("spambase.names") as f:
  for i in range(33):
    f.readline()

  for i in range(57):
    line = f.readline()
    column_names.append(line.split(":")[0])
```

## Train the model

```
model = DecisionTreeClassifier(max_depth=4)
model.fit(data_train, labels_train)
```

## Make predictions for the test data

```
pred = model.predict(data_test)
```

## Calculate the accuracy on the test data

```
np.sum(pred == labels_test)/len(labels_test)
```

## Display the decision tree

```
from six import StringIO
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image

dot_data = StringIO()
export_graphviz(model, out_file=dot_data, feature_names=column_names, \
                    class_names=['not spam', 'spam'], filled=True, rounded=True, \
                    special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
```

![image](https://github.com/user-attachments/assets/336f3af7-2fd3-477f-857f-5d654dc31e5b)

## Closing Remarks

If you want to know more about decision tree algorithms, see the [3rd Chapter of Tan, Steinbach, Karpatne, Kumar: Introduction to Data Mining (Second Edition)](https://www-users.cse.umn.edu/~kumar001/dmbook/ch3_classification.pdf). 

Instead of a decision tree, we could have used many other classifiers from scikit learn, see also [https://scikit-learn.org/1.5/supervised_learning.html](https://scikit-learn.org/1.5/supervised_learning.html).
